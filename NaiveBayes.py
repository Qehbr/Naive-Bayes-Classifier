import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


class NaiveBayesClassifier:
    def __init__(self):
        self.m = 3
        self.folder_path = ""  # folder with files
        self.num_bins = None  # number of bins from user

        self.train_data = None  # train data after filling missing na values and discretization
        self.discretizers = dict()  # discretizer containing bins from training data
        self.col_names = dict()  # dictionary with key: name of column, value: possible attributes
        self.class_probabilities = dict()  # dictionary with key: possible class, value: tuple(probability, n of class)

    """
        Helper function to update num_bins when its updated in GUI
        @param self: NaiveBayesClassifier instance
        @param event: Event of changing in GUI
    """
    def on_bins_entry_change(self, event):
        self.num_bins = event.widget.get()

    """
        Function to get proper folder from the user containing the relevant files
        @param self: NaiveBayesClassifier instance
        @return: Chosen folder
    """
    def browse_folder(self):
        # errors
        folder_path = filedialog.askdirectory()
        if not folder_path:
            messagebox.showerror("Wrong data", "Folder path is empty.")
            return ""
        files = os.listdir(folder_path)
        if len(files) != 3:
            messagebox.showerror("Wrong data", "Number of files in given folder should be 3")
            return ""
        if "test.csv" not in files:
            messagebox.showerror("Wrong data", "test.csv is missing")
            return ""
        if "train.csv" not in files:
            messagebox.showerror("Wrong data", "train.csv is missing")
            return ""
        if "Structure.txt" not in files:
            messagebox.showerror("Wrong data", "Structure.txt is missing")
            return ""

        self.folder_path = folder_path
        return folder_path

    """
        Function to building model including filling missing values, discretization and filling all class variables
        @param self: NaiveBayesClassifier instance
    """
    def build_model(self):
        # Check errors
        if not self.folder_path:
            messagebox.showerror("Wrong data", "Folder path is empty.")
            return ""
        if not self.num_bins:
            messagebox.showerror("Wrong data", "Number of bins is empty")
            return
        if not self.num_bins.isdigit() or int(self.num_bins) <= 1:
            messagebox.showerror("Wrong data", "Number of bins must be integer")
            return

        # Get structure
        with open(os.path.join(self.folder_path, "Structure.txt"), "r") as structure_file:
            structure_lines = structure_file.readlines()
        for line in structure_lines:
            if line.startswith("@ATTRIBUTE"):
                line = line.strip("@ATTRIBUTE").strip()
                # update name of column and possible attributes for column
                if line.endswith("NUMERIC"):
                    col_name = line[:line.index("NUMERIC")].strip()
                    self.col_names[col_name] = "NUMERIC"
                else:
                    col_name = line.split("{", 1)[0].strip()
                    possible_attributes = line.split("{", 1)[-1].rsplit("}", 1)[0].split(",")
                    self.col_names[col_name] = possible_attributes

        # Read train data
        train_data = pd.read_csv(os.path.join(self.folder_path, "train.csv", ), header=0)

        # Fill missing values
        missing_values = train_data.isna().any()
        for column, has_missing_values in missing_values.items():
            if has_missing_values:
                # if it's numeric fill with mean
                if self.col_names[column] == "NUMERIC":
                    train_data[column] = train_data[column].fillna(train_data[column].mean())
                # if it's not numeric with the most frequent value
                else:
                    train_data[column] = train_data[column].fillna(train_data[column].mode().iloc[0])

        # Discretization
        for column in self.col_names:
            if self.col_names[column] == "NUMERIC":
                # make discretizer for given column
                self.discretizers[column] = KBinsDiscretizer(n_bins=int(self.num_bins), encode='ordinal', strategy='uniform')
                # get bins for training data
                train_data[column] = self.discretizers[column].fit_transform(train_data[column].values.reshape(-1, 1))

        # Calculating class probabilities
        for cls in self.col_names['class']:
            n_class = (train_data["class"] == cls).sum()
            self.class_probabilities[cls] = (n_class / len(train_data["class"]), n_class)

        self.train_data = train_data
        messagebox.showinfo("Success!", "Building classifier using train-set is done!")

    """
        Function to classify test_data given train_data
        @param self: NaiveBayesClassifier instance
        @param m: m for m-estimator (currently 2 by default but can be changed)
    """
    def classify_records(self):
        # error if still didn't train
        if self.train_data is None:
            messagebox.showerror("Error!", "You should build the model firstly!")
            return

        # Read test data
        test_data = pd.read_csv(os.path.join(self.folder_path, "test.csv", ), names=self.col_names, header=0)
        # Drop class because we don't want to make predictions based on result :)
        test_data = test_data.drop('class', axis=1)

        # Fill missing values
        missing_values = test_data.isna().any()
        for column, has_missing_values in missing_values.items():
            if has_missing_values:
                # if it's numeric fill with mean !FROM TRAINING DATA!
                if self.col_names[column] == "NUMERIC":
                    test_data[column] = test_data[column].fillna(self.train_data[column].mean())
                # if it's not numeric with the most frequent value !FROM TRAINING DATA!
                else:
                    test_data[column] = test_data[column].fillna(self.train_data[column].mode().iloc[0])

        # Discretization
        for column in self.col_names.keys():
            if self.col_names[column] == "NUMERIC":
                # get bins for test_data using bins from train_data
                test_data[column] = self.discretizers[column].transform(test_data[column].values.reshape(-1, 1))

        with open(os.path.join(self.folder_path, "output.txt"), "w") as file:
            # get each new row from test_data
            for index, row in test_data.iterrows():
                # result dictionary with key: class, value: result from naive bayes with m-estimator
                result = dict()
                for cls in self.class_probabilities:
                    # result containing result for given class, starts with probability for given class
                    cls_result = self.class_probabilities[cls][0]
                    # number of occurrences of class in train_data (used for calculating in naive bayes)
                    n_class = self.class_probabilities[cls][1]
                    # iterate through each feature of row
                    for col in self.col_names:
                        # we don't want to make predictions based on result :)
                        if col == 'class':
                            continue
                        # get value for given row and col
                        value = row[col]
                        # get number of rows containing value of given row and col and class
                        value_and_class = len(
                            self.train_data[(self.train_data[col] == value) & (self.train_data["class"] == cls)])
                        # update result for naive bayes using m-estimator
                        if self.col_names[col] == "NUMERIC":
                            cls_result *= (value_and_class + (self.m * 1 / int(self.num_bins))) / (n_class + self.m)
                        else:
                            cls_result *= (value_and_class + (self.m * 1 / len(self.col_names[col]))) / (n_class + self.m)
                    # save naive bayes result for given class
                    result[cls] = cls_result
                # find class with max naive bayes and write it into file
                max_value = max(result, key=result.get)
                file.write(f"{index + 1} {max_value}\n")
        messagebox.showinfo("Success!", f"Successfully classified and saved to {os.path.join(self.folder_path, 'output.txt')}")

    """
        Function to create GUI
        @param self: NaiveBayesClassifier instance
    """
    def create_gui(self):

        root = tk.Tk()
        root.title("Exercise GUI")

        # Get path to the folder
        path_label = tk.Label(root, text="Path to the folder:")
        path_label.pack()

        path_entry = tk.Entry(root, width=50)
        path_entry.pack()

        browse_button = tk.Button(root, text="Browse", command=lambda: path_entry.insert(tk.END, self.browse_folder()))
        browse_button.pack()

        # Get amount of bins
        bins_label = tk.Label(root, text="Discretization Bins:")
        bins_label.pack()

        bins_entry = tk.Entry(root, width=10)
        bins_entry.pack()

        bins_entry.bind("<KeyRelease>", self.on_bins_entry_change)

        # m for m-estimate (cannot be changed for exercise purposes)
        mestimate_label = tk.Label(root, text="M-Estimate: m = ")
        mestimate_label.pack()

        mestimate_entry = tk.Entry(root, width=10)
        mestimate_entry.insert(0, str(self.m))
        mestimate_entry.configure(state='readonly')
        mestimate_entry.pack()

        # Build model
        build_button = tk.Button(root, text="Build", command=lambda: self.build_model())
        build_button.pack()

        # Classify
        classify_button = tk.Button(root, text="Classify", command=lambda: self.classify_records())
        classify_button.pack()

        root.mainloop()


if __name__ == "__main__":
    nb = NaiveBayesClassifier()
    nb.create_gui()
