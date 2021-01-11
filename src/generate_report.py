import argparse
import datetime

import xlsxwriter
from natsort import natsorted
from tensorflow.compat.v1.train import summary_iterator


def _process_value(value, data, tag, label, num_chunk):
    num_val: float = round(float(value.strip()), 2)

    before_keyword, keyword, after_keyword = tag.partition("Word being trained: ")
    word = after_keyword.split(",")[0]

    before_keyword, keyword, after_keyword = tag.partition(
        " Word Embeddings being used: "
    )
    word_embedding = after_keyword.split(",")[0]

    before_keyword, keyword, after_keyword = tag.partition("Model being used: ")
    num_chunk += 2
    model = after_keyword[:-num_chunk].split("(")[0]

    if word_embedding not in data:
        data[word_embedding] = [{word: {f"{label}-{model}": num_val}}]
    else:
        data[word_embedding].append({word: {f"{label}-{model}": num_val}})
    return data


def _save_to_excel(meta_data, data):
    workbook = xlsxwriter.Workbook(
        fr'../evaluation/report_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.xlsx'
    )
    worksheet = workbook.add_worksheet("Meta Data")

    # Start from the first cell. Rows and
    # columns are zero indexed.
    row = 0
    col = 0

    for m_d in meta_data:
        keyword = after_keyword = ""
        if "Logging started." in m_d:
            before_keyword, keyword, after_keyword = m_d.partition(
                "Logging started. Current date and time:"
            )
        if "Pytorch Version:" in m_d:
            before_keyword, keyword, after_keyword = m_d.partition("Pytorch Version:")
        if "Device:" in m_d:
            before_keyword, keyword, after_keyword = m_d.partition("Device:")
        if "Number of word embeddings being trained:" in m_d:
            before_keyword, keyword, after_keyword = m_d.partition(
                "Number of word embeddings being trained:"
            )
        if "Number of words being trained:" in m_d:
            before_keyword, keyword, after_keyword = m_d.partition(
                "Number of words being trained:"
            )

        worksheet.write(row, col, keyword.strip())
        worksheet.write(row, col + 1, after_keyword.strip())
        row += 1

    worksheet.write(row, col, "Learning Rate")
    worksheet.write(row, col + 1, 0.0001)
    row += 1
    worksheet.write(row, col, "Number of Epochs")
    worksheet.write(row, col + 1, 5000)
    row += 1
    worksheet.write(row, col, "Batch Size")
    worksheet.write(row, col + 1, 100)
    row += 1

    lr_accuracy_worksheet = "Accuracy-LogisticRegression"
    lr_auc_worksheet = "AUC-LogisticRegression"
    ann_accuracy_worksheet = "Accuracy-SingleLayeredNN"
    ann_auc_worksheet = "AUC-SingleLayeredNN"

    worksheets = [
        lr_accuracy_worksheet,
        lr_auc_worksheet,
        ann_accuracy_worksheet,
        ann_auc_worksheet,
    ]
    embeddings = natsorted(list(data.keys()))
    for worksheet_name in worksheets:
        worksheet = workbook.add_worksheet(worksheet_name)
        for index, embedding in enumerate(embeddings, start=1):
            worksheet.write(0, 0, "Hypernym")
            worksheet.write(0, index, embedding)

            tmp_dit = {}
            for x in data[embedding]:
                list_keys = list(x.keys())
                list_values = list(x.values())
                if list_keys[0] not in tmp_dit:
                    tmp_dit[list_keys[0]] = [list_values[0]]
                else:
                    tmp_dit[list_keys[0]].append(list_values[0])
            row = 1
            col = index
            for key, values in tmp_dit.items():
                worksheet.write(row, 0, key)
                for value in values:
                    if list(value.keys())[0] == worksheet_name:
                        worksheet.write(row, col, list(value.values())[0])
                        row += 1

    workbook.close()


def main(script_args):
    meta_data = set()
    data = {}
    for e in summary_iterator(script_args.file_path):
        for v in e.summary.value:
            tag = v.tag
            if "text_summary" in tag:
                value = v.tensor.string_val[0].decode("utf-8")
                if tag == "Description/text_summary":
                    if "Logging started." in value:
                        meta_data.add(value)
                    if "Pytorch Version:" in value:
                        meta_data.add(value)
                    if "Device:" in value:
                        meta_data.add(value)
                    if "Number of word embeddings being trained:" in value:
                        meta_data.add(value)
                    if "Number of words being trained:" in value:
                        meta_data.add(value)
                if "Accuracy/text_summary" == tag[-21:]:
                    data = _process_value(
                        value, data, tag, label="Accuracy", num_chunk=21
                    )
                if "_auc/text_summary" == tag[-17:]:
                    data = _process_value(value, data, tag, label="AUC", num_chunk=17)

    _save_to_excel(meta_data, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        default=r"..\runs\Dec18_17-17-09_LAPTOP-TMI0A1R42020-12-18-62229_epoch_num-5000_verbose-70_word-move_batch_size"
        "-100_all-True_all_embeddings-True_all_models-True_learning_rate-0.0001_word_embeddings-GLOVE_6B_300D"
        "\events.out.tfevents.1608301031.LAPTOP-TMI0A1R4.7900.0",
        help="File Path to read even logs from",
    )
    args = parser.parse_args()
    main(args)
