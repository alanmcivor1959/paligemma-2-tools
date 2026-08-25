import re


def parse_boxes(output_text, img_w, img_h):
    """
    Parses PaliGemma <locXXXX> tokens and scales them to original image pixels.
    Format is: <locYMIN><locXMIN><locYMAX><locXMAX> label
    """
    # Regex pattern to extract location tokens and the trailing label
    pattern = r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([^;?<]+)"

    # Here are some examples of invalid output that was observerd
    #  <loc0000><loc0000><loc1022><loc1022> <seg078><seg078><seg114><seg078><seg078><seg078><seg114><seg078><seg078><seg078><seg114><seg078><seg078><seg078><seg114><seg012> person ;
    #  <loc0000>  (this was right at the end)
    #  <loc0000><loc0410><loc0034><loc0421> person riding  (this was right at the end)
    #  <loc0000><loc0430>  (this was right at the end)
    #  <loc0000><loc0000><loc1023><loc1022> area ;
    #  <loc0000><loc0000><loc1022><loc1022> person on  (this was right at the end)

    matches = re.findall(pattern, output_text)

    results = []
    for match in matches:
        ymin_norm, xmin_norm, ymax_norm, xmax_norm = map(int, match[:4])
        label = match[4].strip()

        # Denormalize from 1024-grid to actual image plane coordinates
        xmin = (xmin_norm / 1024) * img_w
        ymin = (ymin_norm / 1024) * img_h
        xmax = (xmax_norm / 1024) * img_w
        ymax = (ymax_norm / 1024) * img_h

        results.append({"label": label, "box": [xmin, ymin, xmax, ymax]})
    return results
