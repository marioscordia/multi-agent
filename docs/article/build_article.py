#!/usr/bin/env python3
"""Build the journal-style article.

    python docs/article/build_article.py --template path/to/style-template.docx

Replaces the {key} citations in ARTICLE.src.md with numbers in order of first
appearance, appends the numbered reference list, writes ARTICLE.md, and
converts it to dist/ARTICLE.docx with pandoc, taking fonts, heading sizes,
caption styles and page margins from the template document.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

REFS = {
    "own": "Akhmetov M. Safety violation detection in educational institutions: A systematic bibliometric review of deep learning and computer vision approaches. 2026 IEEE 3rd International Student Conference on Digital Generation, Astana, Kazakhstan, 2026, paper ID-22.",
    "perseghin": "Perseghin E., Foresti G. L. A shallow system prototype for violent action detection in Italian public schools. Information, 2023, vol. 14, no. 4, art. 240. DOI: 10.3390/info14040240",
    "kozham": "Kozhamkulova Z., Kirgizbayeva B., Sembina G., Smailova U., Suleimenova M., Keneskanova A., Baizakova Z. MoveNET enabled neural network for fast detection of physical bullying in educational institutions. International Journal of Advanced Computer Science and Applications, 2023, vol. 14, no. 5. DOI: 10.14569/IJACSA.2023.0140578",
    "omarov": "Omarov B., Narynov S., Zhumanov Z., Gumar A., Khassanova M. A skeleton-based approach for campus violence detection. Computers, Materials & Continua, 2022. DOI: 10.32604/cmc.2022.024566",
    "guo": "Guo T., Chen X., Wang Y., Chang R., Pei S., Chawla N. V., Wiest O., Zhang X. Large language model based multi-agents: A survey of progress and challenges. Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI-24), 2024.",
    "wu": "Wu Q., Bansal G., Zhang J., Wu Y., Li B., Zhu E., Jiang L., Zhang X., Zhang S., Liu J., Awadallah A. H., White R. W., Burger D., Wang C. AutoGen: Enabling next-gen LLM applications via multi-agent conversation. arXiv preprint arXiv:2308.08155, 2023.",
    "hong": "Hong S., Zhuge M., Chen J., Zheng X., Cheng Y., Zhang C., Wang J., Wang Z., Yau S. K. S., Lin Z., Zhou L., Ran C., Xiao L., Wu C., Schmidhuber J. MetaGPT: Meta programming for a multi-agent collaborative framework. International Conference on Learning Representations (ICLR), 2024.",
    "wooldridge": "Wooldridge M. An Introduction to MultiAgent Systems. 2nd ed. John Wiley & Sons, Chichester, 2009.",
    "hayesroth": "Hayes-Roth B. A blackboard architecture for control. Artificial Intelligence, 1985, vol. 26, no. 3, pp. 251-321. DOI: 10.1016/0004-3702(85)90063-3",
    "hockey": "Bermejo Nievas E., Deniz Suarez O., Bueno Garcia G., Sukthankar R. Violence detection in video using computer vision techniques. Computer Analysis of Images and Patterns (CAIP 2011), Lecture Notes in Computer Science, vol. 6855, Springer, 2011, pp. 332-339.",
    "sultani": "Sultani W., Chen C., Shah M. Real-world anomaly detection in surveillance videos. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 6479-6488.",
    "rwf": "Cheng M., Cai K., Li M. RWF-2000: An open large scale video database for violence detection. Proceedings of the 25th International Conference on Pattern Recognition (ICPR), 2021.",
    "ha": "Ha N. D., Tran N. Y., Thuy L. N. L., Shimizu I., Bao P. T. Violence region localization in video and the school violent actions classification. Frontiers in Computer Science, 2023, vol. 5. DOI: 10.3389/fcomp.2023.1274928",
    "garcia": "Garcia-Cobo G., SanMiguel J. C. Human skeletons and change detection for efficient violence detection in surveillance videos. Computer Vision and Image Understanding, 2023, art. 103739. DOI: 10.1016/j.cviu.2023.103739",
    "qi": "Qi B., Wu B., Sun B. Automated violence monitoring system for real-time fistfight detection using deep learning-based temporal action localization. Scientific Reports, 2025, vol. 15. DOI: 10.1038/s41598-025-12531-4",
    "ali": "Ali S., Islam M. T., Lee I. H., Hijji M., Muhammad K. CABAD: A video dataset for benchmarking child aggression recognition. Alexandria Engineering Journal, 2025. DOI: 10.1016/j.aej.2025.06.035",
    "tapia": "Tapia Leon F., Collahuazo J. D., Farinango P. C., Chiliquinga C., Tello-Oquendo L., Rivera J. Early warning system for firearm detection on university campuses using computer vision. Journal of Internet Services and Information Security, 2026. DOI: 10.58346/jisis.2026.i1.008",
    "berardini": "Berardini D., Migliorelli L., Galdelli A., Frontoni E., Mancini A., Moccia S. A deep-learning framework running on edge devices for handgun and knife detection from indoor video-surveillance cameras. Multimedia Tools and Applications, 2023. DOI: 10.1007/s11042-023-16231-x",
    "dasilva": "Da Silva A. V. B., Pereira L. F. A. Evaluating methods for violence classification and firearm detection in indoor CCTV environment. Journal of the Brazilian Computer Society, 2024. DOI: 10.5753/jbcs.2024.3282",
    "yolo": "Redmon J., Divvala S., Girshick R., Farhadi A. You only look once: Unified, real-time object detection. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 779-788.",
    "yolov8": "Jocher G., Chaurasia A., Qiu J. Ultralytics YOLOv8. Software, version 8, 2023. Available at: https://github.com/ultralytics/ultralytics",
    "coco": "Lin T.-Y., Maire M., Belongie S., Hays J., Perona P., Ramanan D., Dollar P., Zitnick C. L. Microsoft COCO: Common objects in context. Computer Vision - ECCV 2014, Lecture Notes in Computer Science, vol. 8693, Springer, 2014, pp. 740-755. DOI: 10.1007/978-3-319-10602-1_48",
    "yao": "Yao S., Zhao J., Yu D., Du N., Shafran I., Narasimhan K., Cao Y. ReAct: Synergizing reasoning and acting in language models. International Conference on Learning Representations (ICLR), 2023.",
    "huang": "Huang J., Chen X., Mishra S., Zheng H. S., Yu A. W., Song X., Zhou D. Large language models cannot self-correct reasoning yet. International Conference on Learning Representations (ICLR), 2024.",
    "du": "Du Y., Li S., Torralba A., Tenenbaum J. B., Mordatch I. Improving factuality and reasoning in language models through multiagent debate. Proceedings of the 41st International Conference on Machine Learning, PMLR, 2024, vol. 235.",
    "anthropic": "Anthropic. Structured outputs. Claude API documentation, 2026. Available at: https://platform.claude.com/docs",
    "sort": "Bewley A., Ge Z., Ott L., Ramos F., Upcroft B. Simple online and realtime tracking. Proceedings of the IEEE International Conference on Image Processing (ICIP), 2016, pp. 3464-3468. DOI: 10.1109/ICIP.2016.7533003",
    "bm25": "Robertson S., Zaragoza H. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 2009, vol. 3, no. 4, pp. 333-389. DOI: 10.1561/1500000019",
}


def number_citations(text: str) -> tuple[str, list[str]]:
    order: list[str] = []

    def num(key: str) -> int:
        if key not in REFS:
            raise SystemExit(f"unknown citation key: {key}")
        if key not in order:
            order.append(key)
        return order.index(key) + 1

    # adjacent citations "{a} {b}" become one bracket "[1, 2]"
    def group(match: re.Match) -> str:
        keys = re.findall(r"\{(\w+)\}", match.group(0))
        return "[" + ", ".join(str(num(k)) for k in keys) + "]"

    # only known keys count as citations, so LaTeX such as \mathbf{x} is left alone
    known = "|".join(sorted(map(re.escape, REFS), key=len, reverse=True))
    cite = rf"\{{(?:{known})\}}"
    body = re.sub(rf"{cite}(?:\s{cite})*", group, text)
    return body, order


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, help="docx whose styles and page setup to copy")
    parser.add_argument("-o", "--output", default=str(REPO / "dist" / "ARTICLE.docx"))
    args = parser.parse_args()

    src = (HERE / "ARTICLE.src.md").read_text(encoding="utf-8")
    # protect pandoc's {custom-style=...} and {width=...} attributes from the citation pass
    protected = re.sub(r"\{(custom-style|width)=", r"⟦\1=", src)
    body, order = number_citations(protected)
    body = body.replace("⟦", "{")

    # "1\\." keeps the number as text, as in the template, instead of a Word list
    refs = "\n".join(
        f'::: {{custom-style="Reference Item"}}\n{i}\\. {REFS[k]}\n:::\n' for i, k in enumerate(order, 1)
    )
    unused = sorted(set(REFS) - set(order))
    if unused:
        print("note: references never cited:", ", ".join(unused))
    out_md = HERE / "ARTICLE.md"
    out_md.write_text(body.rstrip() + "\n\n" + refs, encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        reference = _reference_from_template(Path(args.template), Path(tmp))
        subprocess.run(
            ["pandoc", str(out_md), "-o", args.output, "--from=markdown-auto_identifiers",
             f"--reference-doc={reference}", f"--resource-path={HERE}"],
            check=True,
        )
    _declare_png(Path(args.output))
    print(f"  wrote {out_md.relative_to(REPO)} and {args.output} ({len(order)} references)")
    return 0


def _reference_from_template(template: Path, tmp: Path) -> Path:
    """A clean pandoc reference document carrying only the template's styles,
    theme and page setup. Using the template itself as the reference would copy
    its pictures into the output as unregistered parts."""
    import zipfile

    base = tmp / "default.docx"
    with base.open("wb") as fh:
        subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                       stdout=fh, check=True)
    with zipfile.ZipFile(template) as tz:
        styles = tz.read("word/styles.xml").decode("utf-8")
        theme = tz.read("word/theme/theme1.xml")
        sect = re.findall(r"<w:sectPr\b.*?</w:sectPr>", tz.read("word/document.xml").decode("utf-8"), re.S)[-1]
    sect = re.sub(r' w:rsid\w*="[^"]*"', "", sect)
    # the author line is right-aligned and italic in the template
    styles = re.sub(
        r'(<w:style [^>]*w:styleId="Author".*?<w:pPr>)(.*?)(</w:pPr>)',
        r'\1<w:ind w:firstLine="0"/><w:jc w:val="right"/>\3', styles, count=1, flags=re.S)
    styles = re.sub(
        r'(<w:style [^>]*w:styleId="Author".*?)(</w:style>)',
        lambda m: m.group(1) + ("" if "<w:rPr>" in m.group(1) else "<w:rPr><w:i/></w:rPr>") + m.group(2),
        styles, count=1, flags=re.S)

    out = tmp / "reference.docx"
    with zipfile.ZipFile(base) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item)
            if item.filename == "word/styles.xml":
                data = styles.encode("utf-8")
            elif item.filename == "word/theme/theme1.xml":
                data = theme
            elif item.filename == "word/document.xml":
                data = re.sub(r"<w:sectPr\b.*?</w:sectPr>", sect, data.decode("utf-8"), flags=re.S).encode("utf-8")
            zout.writestr(item, data)
    return out


def _declare_png(docx: Path) -> None:
    """Pandoc registers each image by part name only; Word's validator also
    expects a default content type for the .png extension. Add it if missing."""
    import shutil
    import tempfile
    import zipfile

    with zipfile.ZipFile(docx) as zin:
        types = zin.read("[Content_Types].xml").decode("utf-8")
        if 'Extension="png"' in types:
            return
        types = types.replace(
            "<Default ", '<Default Extension="png" ContentType="image/png"/><Default ', 1
        )
        tmp = Path(tempfile.mkstemp(suffix=".docx")[1])
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename.endswith("/"):
                    continue
                data = types.encode("utf-8") if item.filename == "[Content_Types].xml" else zin.read(item)
                zout.writestr(item, data)
    shutil.move(tmp, docx)


if __name__ == "__main__":
    raise SystemExit(main())
