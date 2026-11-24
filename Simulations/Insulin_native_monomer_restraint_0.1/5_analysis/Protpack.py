#!/usr/bin/env python3
"""protpack: packs the protonation records of a constant-pH run for archiving, and unpacks them again.

Put this script in the directory with the reordered_cpouts.pH_* files and run it without arguments.
It looks at every window and does what is missing:

  reordered_cpouts.pH_7.00 exists, reordered_cpouts.pH_7.00.protpack does not  ->  the packed file is written
  the packed file exists, the plain file does not                               ->  the plain file is restored
  both exist                                                                    ->  nothing (or compared, with --check)

Nothing is ever deleted. After packing, run with --check and delete the plain files yourself once every
window reports "match".

    python protpack.py                 # in the script's own directory
    python protpack.py /path/to/dir    # elsewhere
    python protpack.py --check         # compare the plain and the packed file of every window that has both
    python protpack.py --jobs 8        # parallel workers (default: up to 4)

How a file is packed. A cpout line is a template plus numbers, "Residue {} State: {}", so the file is
taken apart into one template id per line and one column of integers per number field. The columns are
delta-coded and compressed with xz. The frame counters become constant deltas and the protonation states
a sparse stream of changes, which brings a window down by a factor of several hundred. Numbers are stored
as the integer behind their decimal digits with the field width, never as floats, so 0.200 comes back as
0.200. A line that would not come back identical (007, -0.0, 1.0E-05) is kept verbatim. Before a packed
file is written, the plain file is regenerated from the columns and compared by sha256; the sha256 is
kept in the packed file and checked again when it is restored, so a damaged file cannot be restored
silently. A file that is not line-and-number text is stored as plain xz inside the same container.

Needs Python 3 and numpy, nothing else.
"""
import argparse
import array
import glob
import hashlib
import json
import lzma
import os
import re
import stat
import struct
import sys
import time
from multiprocessing import Pool

import numpy as np

__version__ = "2.0"

MAGIC = b"PROTPACK\n"
PATTERN = "reordered_cpouts.pH_*"
SUFFIX = ".protpack"
NUM_RE = re.compile(r"[ \t]*-?\d+(?:\.\d+)?")
CHUNK_BYTES = 32 * 1024 * 1024          # text read per step while packing
CHUNK_LINES = 1_000_000                 # lines regenerated per step while unpacking
LINE_CACHE_MAX = 200_000                # distinct lines remembered per file
VERBATIM = 0                            # template id of a line kept as text


# ----------------------------------------------------------------------------
# numbers and lines
# ----------------------------------------------------------------------------
def fmt_num(mant, dec, width):
    """The text of a number from its integer mantissa, decimals and field width."""
    if dec:
        s = str(abs(mant)).rjust(dec + 1, "0")
        s = s[:-dec] + "." + s[-dec:]
        if mant < 0:
            s = "-" + s
    else:
        s = str(mant)
    return s.rjust(width)


def encode_line(line):
    """(template parts, widths, decimals, mantissas) of a line, or None if it would not come back identical."""
    parts, widths, decs, vals = [], [], [], []
    pos = 0
    for m in NUM_RE.finditer(line):
        tok = m.group()
        s = tok.lstrip(" \t")
        if "." in s:
            ip, fp = s.split(".")
            dec = len(fp)
            mant = int(ip + fp)
        else:
            dec = 0
            mant = int(s)
        if abs(mant) >= (1 << 62) or fmt_num(mant, dec, len(tok)) != tok:
            return None
        parts.append(line[pos:m.start()])
        widths.append(len(tok))
        decs.append(dec)
        vals.append(mant)
        pos = m.end()
    parts.append(line[pos:])
    return tuple(parts), tuple(widths), tuple(decs), vals


def decode_line(parts, widths, decs, vals):
    out = [parts[0]]
    for i, v in enumerate(vals):
        out.append(fmt_num(v, decs[i], widths[i]))
        out.append(parts[i + 1])
    return "".join(out)


def pack_ints(values):
    """Delta-code an integer sequence and store it in the narrowest dtype that holds the deltas."""
    a = np.asarray(values, dtype=np.int64)
    if a.size == 0:
        return "int8", b""
    d = np.diff(a, prepend=np.int64(0))
    lim = np.abs(d).max()
    for dtype in ("int8", "int16", "int32", "int64"):
        if lim < np.iinfo(dtype).max:
            break
    return dtype, d.astype(dtype).tobytes()


def unpack_ints(dtype, raw):
    return np.cumsum(np.frombuffer(raw, dtype=dtype), dtype=np.int64)


# ----------------------------------------------------------------------------
# the column codec
# ----------------------------------------------------------------------------
class ColumnWriter:
    """Takes the lines of a text file apart into templates, template ids and integer columns."""

    def __init__(self):
        self.shapes = {}                # key -> template id;  key = (parts, widths, decs)
        self.shape_list = [None]        # id -> key, id 0 is VERBATIM
        self.columns = [[]]             # id -> [array('q') per field]
        self.sids = array.array("H")
        self.verbatim = []
        self.cache = {}

    def add(self, line):
        hit = self.cache.get(line)
        if hit is not None:
            sid, vals = hit
            self.sids.append(sid)
            cols = self.columns[sid]
            for i, v in enumerate(vals):
                cols[i].append(v)
            return
        enc = encode_line(line)
        if enc is None:
            self.sids.append(VERBATIM)
            self.verbatim.append(line)
            return
        parts, widths, decs, vals = enc
        key = (parts, widths, decs)
        sid = self.shapes.get(key)
        if sid is None:
            if len(self.shape_list) >= 65535:
                self.sids.append(VERBATIM)
                self.verbatim.append(line)
                return
            sid = len(self.shape_list)
            self.shapes[key] = sid
            self.shape_list.append(key)
            self.columns.append([array.array("q") for _ in widths])
        self.sids.append(sid)
        cols = self.columns[sid]
        for i, v in enumerate(vals):
            cols[i].append(v)
        if len(self.cache) < LINE_CACHE_MAX:
            self.cache[line] = (sid, vals)

    def header_and_blobs(self, ends_with_newline):
        header = {"n_lines": len(self.sids), "ends_with_newline": ends_with_newline,
                  "shapes": [], "verbatim": self.verbatim, "columns": []}
        blobs = [np.frombuffer(self.sids, dtype=np.uint16).tobytes()]
        header["sids_nbytes"] = len(blobs[0])
        for sid in range(1, len(self.shape_list)):
            parts, widths, decs = self.shape_list[sid]
            header["shapes"].append({"parts": list(parts), "widths": list(widths), "decs": list(decs)})
            for col in self.columns[sid]:
                dtype, raw = pack_ints(np.frombuffer(col, dtype=np.int64) if len(col) else [])
                header["columns"].append({"sid": sid, "dtype": dtype, "nbytes": len(raw)})
                blobs.append(raw)
        return header, blobs


def iter_lines(fh, sha):
    """Yield (lines without their newline, whether the chunk ended with a newline), latin-1 decoded."""
    rest = b""
    while True:
        chunk = fh.read(CHUNK_BYTES)
        if not chunk:
            break
        sha.update(chunk)
        data = rest + chunk
        cut = data.rfind(b"\n")
        if cut < 0:
            rest = data
            continue
        yield data[:cut].decode("latin-1").split("\n"), True
        rest = data[cut + 1:]
    if rest:
        yield rest.decode("latin-1").split("\n"), False


def columns_encode(path):
    """(header, blobs, sha256) of a text file."""
    writer = ColumnWriter()
    sha = hashlib.sha256()
    ends_with_newline = False
    with open(path, "rb") as fh:
        for lines, nl in iter_lines(fh, sha):
            for line in lines:
                writer.add(line)
            ends_with_newline = nl
    header, blobs = writer.header_and_blobs(ends_with_newline)
    return header, blobs, sha.hexdigest()


def columns_decode(header, data, pos, out):
    """Regenerate the text from the columns starting at data[pos] into the binary stream `out`."""
    n = header["n_lines"]
    sids = np.frombuffer(data[pos:pos + header["sids_nbytes"]], dtype=np.uint16)
    pos += header["sids_nbytes"]
    shapes = [None] + [(tuple(s["parts"]), tuple(s["widths"]), tuple(s["decs"])) for s in header["shapes"]]
    columns = [[] for _ in shapes]
    for c in header["columns"]:
        columns[c["sid"]].append(unpack_ints(c["dtype"], data[pos:pos + c["nbytes"]]))
        pos += c["nbytes"]
    verbatim = header["verbatim"]
    cursor = [0] * len(shapes)
    vcursor = 0
    cache = {}
    for start in range(0, n, CHUNK_LINES):
        stop = min(n, start + CHUNK_LINES)
        chunk = sids[start:stop]
        out_lines = [None] * (stop - start)
        for sid in np.unique(chunk):
            sid = int(sid)
            idx = np.flatnonzero(chunk == sid)
            k = len(idx)
            if sid == VERBATIM:
                for j, i in enumerate(idx.tolist()):
                    out_lines[i] = verbatim[vcursor + j]
                vcursor += k
                continue
            parts, widths, decs = shapes[sid]
            cols = [col[cursor[sid]:cursor[sid] + k].tolist() for col in columns[sid]]
            cursor[sid] += k
            for j, i in enumerate(idx.tolist()):
                vals = tuple(col[j] for col in cols)
                key = (sid, vals)
                line = cache.get(key)
                if line is None:
                    line = decode_line(parts, widths, decs, vals)
                    if len(cache) < LINE_CACHE_MAX:
                        cache[key] = line
                out_lines[i] = line
        text = "\n".join(out_lines)
        if stop < n or header["ends_with_newline"]:
            text += "\n"
        out.write(text.encode("latin-1"))


def looks_columnar(path, sample=1_000_000, max_verbatim=0.2):
    """True if the first lines of the file take apart into templates and numbers."""
    with open(path, "rb") as fh:
        head = fh.read(sample)
    if b"\x00" in head:
        return False
    lines = head.split(b"\n")[:-1]
    if not lines:
        return False
    verbatim = sum(encode_line(line.decode("latin-1")) is None for line in lines)
    return verbatim <= max_verbatim * len(lines)


# ----------------------------------------------------------------------------
# the packed file: xz( MAGIC, header length, header JSON, payload )
# ----------------------------------------------------------------------------
class HashingSink:
    def __init__(self, fh=None):
        self.fh = fh
        self.sha = hashlib.sha256()
        self.size = 0

    def write(self, data):
        self.sha.update(data)
        self.size += len(data)
        if self.fh is not None:
            self.fh.write(data)


def sha256_of(path):
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_BYTES), b""):
            sha.update(chunk)
    return sha.hexdigest()


def regenerate(header, data, pos, out):
    if header["codec"] == "columns":
        columns_decode(header, data, pos, out)
    else:
        out.write(data[pos:])


def write_packed(src, dst, level, verify=True):
    """Pack `src` into `dst`. Returns the header."""
    st = os.stat(src)
    header = {"format": "protpack", "version": __version__, "name": os.path.basename(src), "size": st.st_size,
              "mode": stat.S_IMODE(st.st_mode), "mtime": st.st_mtime}
    if st.st_size > 0 and looks_columnar(src):
        cols, blobs, sha = columns_encode(src)
        header.update(cols)
        header["codec"] = "columns"
    else:
        with open(src, "rb") as fh:
            blobs = [fh.read()]
        sha = hashlib.sha256(blobs[0]).hexdigest()
        header["codec"] = "raw"
    header["sha256"] = sha
    hdr = json.dumps(header, ensure_ascii=False).encode("utf-8")
    data = b"".join([MAGIC, struct.pack("<Q", len(hdr)), hdr] + blobs)
    del blobs
    if verify:
        sink = HashingSink()
        regenerate(header, data, len(MAGIC) + 8 + len(hdr), sink)
        if sink.sha.hexdigest() != sha or sink.size != st.st_size:
            raise RuntimeError(f"{os.path.basename(src)}: the packed form does not reproduce the file, this is a bug")
    tmp = dst + ".partial"
    try:
        with open(tmp, "wb") as fh:
            fh.write(lzma.compress(data, preset=level))
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return header


def read_packed(path):
    """(header, data, position of the payload) of a packed file."""
    try:
        with open(path, "rb") as fh:
            data = lzma.decompress(fh.read())
        if not data.startswith(MAGIC):
            raise ValueError("wrong magic")
        (hlen,) = struct.unpack("<Q", data[len(MAGIC):len(MAGIC) + 8])
        header = json.loads(data[len(MAGIC) + 8:len(MAGIC) + 8 + hlen].decode("utf-8"))
        if header.get("format") != "protpack":
            raise ValueError("not a protpack header")
    except (lzma.LZMAError, ValueError, struct.error, KeyError) as exc:
        raise RuntimeError(f"{os.path.basename(path)}: not a readable protpack file ({exc})") from None
    return header, data, len(MAGIC) + 8 + hlen


def restore(packed, dst):
    """Regenerate the plain file from `packed` into `dst`, checking the sha256. Returns the header."""
    header, data, pos = read_packed(packed)
    tmp = dst + ".partial"
    try:
        with open(tmp, "wb") as fh:
            sink = HashingSink(fh)
            try:
                regenerate(header, data, pos, sink)
            except (ValueError, KeyError, IndexError, struct.error) as exc:
                raise RuntimeError(f"{os.path.basename(packed)}: damaged ({exc})") from None
        if sink.sha.hexdigest() != header["sha256"] or sink.size != header["size"]:
            raise RuntimeError(f"{os.path.basename(packed)}: the restored file does not match its sha256, "
                               f"the packed file is damaged")
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.chmod(dst, header["mode"])
    os.utime(dst, (header["mtime"], header["mtime"]))
    return header


# ----------------------------------------------------------------------------
# workers and the main loop
# ----------------------------------------------------------------------------
def human(n):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} B" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024


def do_pack(args):
    plain, packed, level, verify = args
    t0 = time.time()
    header = write_packed(plain, packed, level, verify)
    size, stored = header["size"], os.path.getsize(packed)
    return (f"packed    {os.path.basename(plain):<32s} {human(size):>10s} -> {human(stored):>10s}  "
            f"x{size / max(1, stored):>7.1f}  {header['codec']:<7s} {time.time() - t0:>6.1f} s")


def do_unpack(args):
    plain, packed = args
    t0 = time.time()
    header = restore(packed, plain)
    return f"restored  {os.path.basename(plain):<32s} {human(header['size']):>10s}  sha256 ok  {time.time() - t0:>6.1f} s"


def do_check(args):
    plain, packed = args
    header, data, pos = read_packed(packed)
    sink = HashingSink()
    regenerate(header, data, pos, sink)
    packed_sha = sink.sha.hexdigest()
    plain_sha = sha256_of(plain)
    if packed_sha != header["sha256"]:
        return f"DAMAGED   {os.path.basename(packed)}: does not regenerate to its own sha256"
    if plain_sha != packed_sha:
        return f"DIFFERENT {os.path.basename(plain)}: the plain file is not the one that was packed"
    return f"match     {os.path.basename(plain):<32s} {human(header['size']):>10s}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", nargs="?", default=os.path.dirname(os.path.abspath(__file__)),
                   help="directory with the records (default: the directory of this script)")
    p.add_argument("--check", action="store_true", help="compare the plain and the packed file where both exist")
    p.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)), help="parallel workers")
    p.add_argument("--level", type=int, default=9, choices=range(1, 10), help="xz preset (default 9)")
    p.add_argument("--no-verify", action="store_true", help="do not regenerate each file from its columns before writing")
    a = p.parse_args(argv)

    d = os.path.abspath(a.dir)
    names = {os.path.basename(f) for f in glob.glob(os.path.join(d, PATTERN))}
    plain = {n for n in names if not n.endswith(SUFFIX)}
    packed = {n[:-len(SUFFIX)] for n in names if n.endswith(SUFFIX)}
    to_pack = sorted(plain - packed)
    to_unpack = sorted(packed - plain)
    both = sorted(plain & packed)
    if not names:
        sys.exit(f"no {PATTERN} files in {d}")
    print(f"{d}: {len(to_pack)} to pack, {len(to_unpack)} to restore, {len(both)} present in both forms")

    jobs = [(do_pack, (os.path.join(d, n), os.path.join(d, n + SUFFIX), a.level, not a.no_verify)) for n in to_pack]
    jobs += [(do_unpack, (os.path.join(d, n), os.path.join(d, n + SUFFIX))) for n in to_unpack]
    if a.check:
        jobs += [(do_check, (os.path.join(d, n), os.path.join(d, n + SUFFIX))) for n in both]
    if not jobs:
        print("nothing to do" + ("" if a.check or not both else ", run with --check to compare the two forms"))
        return
    t0 = time.time()
    failed = 0
    with Pool(a.jobs) as pool:
        for line in pool.imap(run_job, jobs):
            print("  " + line, flush=True)
            failed += line.startswith(("DAMAGED", "DIFFERENT", "FAILED"))
    print(f"{len(jobs)} files in {time.time() - t0:.0f} s" + (f", {failed} FAILED" if failed else ""))
    sys.exit(1 if failed else 0)


def run_job(job):
    func, args = job
    try:
        return func(args)
    except RuntimeError as exc:
        return f"FAILED    {exc}"


if __name__ == "__main__":
    main()
