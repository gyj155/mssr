# Publishing the MSSR research page

The static site in this directory is prepared for `https://gyj155.github.io/mssr/`.

1. Merge the documentation change into `main`.
2. In repository Settings → Pages, select deployment from the `main` branch and `/docs` folder. No build dependencies or JavaScript are needed.
3. Verify the project page, `citation.bib`, images and `sitemap.xml` return HTTP 200.
4. Once live, change the README's `Research overview` link from `docs/index.html` to the public project URL. Add that URL to the repository About section and the author's existing homepage.
