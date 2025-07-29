import fitz  # PyMuPDF

def get_line_and_col(page, annot_rect):
    # Get all words on the page (includes their position)
    words = page.get_text("words")  # list of (x0, y0, x1, y1, word, block_no, line_no, word_no)
    lines = sorted(set(word[1] for word in words))  # y0 of each word = line position

    # Find closest line (based on vertical position of annotation)
    y_center = (annot_rect.y0 + annot_rect.y1) / 2
    line_num = min(range(len(lines)), key=lambda i: abs(lines[i] - y_center)) + 1

    # For column: compare x-coordinate to words on that line
    line_words = [w for w in words if abs(w[1] - lines[line_num - 1]) < 2]
    line_words_sorted = sorted(line_words, key=lambda w: w[0])  # sort by x0

    x_center = (annot_rect.x0 + annot_rect.x1) / 2
    col_num = 1
    for i, word in enumerate(line_words_sorted):
        if word[0] > x_center:
            break
        col_num = i + 1

    return line_num, col_num

doc = fitz.open("example.pdf")

for page_num in range(len(doc)):
    page = doc[page_num]
    for annot in page.annots():
        rect = annot.rect
        content = annot.info.get('content', '')
        line, col = get_line_and_col(page, rect)
        print(f"Page {page_num + 1}, Line {line}, Col {col}: {content}")
