# Atlas-DeePC website

This is a minimal local research website with a paper title, navigation,
overview figures, and sections ready for more content.

Edit `dist/index.md` to change the website:

- The `#` heading sets the fallback browser title.
- `## Site settings` contains the browser title and mobile menu label.
- The links under `## Navigation` define every navigation item and visible
  section. Changing a label changes it in both places; changing the `#target`
  changes the section anchor.
- Each matching `##` heading supplies that section's content. Write paragraphs
  and images using `![Description](assets/figures/image.png)`, separated by blank
  lines. Keep the section heading identical to its navigation label.
- Edit `**Paper title:**` under Site settings to change the centered paper title.

Overview images are browser previews of the originals in `website/figures/`.
Click an image on the page to open it at full size.

`dist/index.html`, `dist/styles.css`, and `dist/app.js` provide the small display
shell. The old video files remain under `dist/assets/` but are not displayed.
The page uses bundled Source Sans 3 fonts: regular body text and semibold headings. Figure
labels retain the fonts embedded in the original paper figures.

## Local preview

From the project root, run:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory website/dist
```

Then open http://localhost:8000.
