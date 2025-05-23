from textblob import TextBlob

def correct_query(query: str) -> (str, bool):
    blob = TextBlob(query)
    corrected_blob = blob.correct()
    corrected_str = str(corrected_blob)

    iscorrected = (corrected_str != query)

    return corrected_str, iscorrected

# Example usage
if __name__ == "__main__":
    query = "Effctive traatments for skinn condtions"
    corrected, iscorrected = correct_query(query)
    if iscorrected:
        print("Do you mean:", corrected)
    print("Original Query:", query)
    print("Corrected Query:", corrected)
