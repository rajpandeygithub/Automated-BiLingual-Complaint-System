import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tabulate import tabulate


def load_data_from_csv(file_path):
    """
    Load data from a local CSV file.
    """
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully from {file_path}. Shape: {data.shape}")
    return data


def extract_sensitive_keywords(data, text_column, label_column, top_n=10):
    """
    Extract sensitive keywords per label using TF-IDF Vectorization.
    """
    data = data.dropna(subset=[text_column, label_column])
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data[text_column])
    keywords_per_label = {}

    for label in data[label_column].unique():
        label_indices = data[label_column] == label
        label_tfidf_matrix = tfidf_matrix[label_indices]
        tfidf_mean = label_tfidf_matrix.mean(axis=0).A1
        top_indices = tfidf_mean.argsort()[-top_n:][::-1]
        top_keywords = [(tfidf_vectorizer.get_feature_names_out()[i], tfidf_mean[i]) for i in top_indices]
        keywords_per_label[label] = top_keywords

    return keywords_per_label


def print_keywords_table(keywords_dict, label_name):
    """
    Print sensitive keywords in a tabular format.
    """
    rows = []
    for label, keywords in keywords_dict.items():
        for keyword, score in keywords:
            rows.append([label, keyword, round(score, 4)])
    print(f"\nSensitive Keywords for {label_name}:")
    print(tabulate(rows, headers=["Label", "Keyword", "Importance Score"], tablefmt="grid"))


def save_keywords_to_csv(keywords_dict, label_name, output_path):
    """
    Save sensitive keywords to a CSV file.
    """
    rows = []
    for label, keywords in keywords_dict.items():
        for keyword, score in keywords:
            rows.append({"Label": label, "Keyword": keyword, "Importance Score": round(score, 4)})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Sensitive keywords for {label_name} saved to {output_path}")


def main():
    # File paths for input data
    preprocessed_data_path = "/Users/archie/Downloads/Untitled spreadsheet - preprocessed_data.csv"
    drift_records_path = "/Users/archie/Downloads/Untitled spreadsheet - drift_records.csv"

    # Load data
    preprocessed_data = load_data_from_csv(preprocessed_data_path)
    drift_records = load_data_from_csv(drift_records_path)

    # Extract sensitive keywords for departments
    department_keywords = extract_sensitive_keywords(preprocessed_data, "complaint", "department")
    print_keywords_table(department_keywords, "Departments")
    save_keywords_to_csv(department_keywords, "Departments", "department_sensitive_keywords.csv")

    # Extract sensitive keywords for products
    product_keywords = extract_sensitive_keywords(preprocessed_data, "complaint", "product")
    print_keywords_table(product_keywords, "Products")
    save_keywords_to_csv(product_keywords, "Products", "product_sensitive_keywords.csv")


if __name__ == "__main__":
    main()
