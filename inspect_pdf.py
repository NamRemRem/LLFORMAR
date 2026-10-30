import pymupdf

doc = pymupdf.open(r'c:\Users\Admin\Documents\LLFormer\LLFormer_Reproduction_Paper.pdf')
print('Page count:', len(doc))
for i, page in enumerate(doc):
    rects = [b[:4] for b in page.get_text('blocks')]
    if rects:
        min_y = min(r[1] for r in rects)
        max_y = max(r[3] for r in rects)
        print(f'Page {i+1}: min_y={min_y:.1f}, max_y={max_y:.1f}, span={max_y-min_y:.1f} / {page.rect.height:.1f}')
