# PRIME logo assets

The mark is a P with a prime, read "P prime". Ink #1B2430, orange #F0603A.
All SVGs have transparent backgrounds and no font dependencies.

* `banner.svg`, `banner-dark.svg`   mark, wordmark and tagline, for the top of the README
* `logo.svg`, `logo-dark.svg`       mark and wordmark, for docs and slides
* `icon.svg`, `icon-white.svg`      the mark alone, for light and dark backgrounds
* `icon-512.png`                    the mark as a raster, for PyPI, Zenodo and social cards
* `banner.png`                      the banner as a raster, for places that do not take SVG

README header, switching with the reader's colour scheme:

    <p align="center">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
        <img src="assets/banner.svg" alt="PRIME: Protonation-number Reweighting in Multi-pH Ensembles" width="720">
      </picture>
    </p>
