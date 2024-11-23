import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV


IMAGE_DIR = 'data/images'

def load_image(image_name, image_class):
    """ Load image by name and class """
    image_path = os.path.join(IMAGE_DIR, image_class, image_name)
    if os.path.exists(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load {image_path} correctly.")
            return None
        return image
    else:
        print(f"Image {image_name} not found in {image_path}")
        return None
    

def show_image(image):
    """ Show image and optionally save to disk"""
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


def load_data():
    """ Load all data """
    df1 = pd.read_csv('data/Data.csv')
    df2 = pd.read_csv('data/extra_hard_samples.csv')
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    return df

def split_data(df):
    """ Split data into X/y training and testing sets from `amount` of total data """
    X = df.drop(['image_name', 'class'], axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def validation_scores(pipe, param_grid, X_train, y_train):
    """ Create, run, and return cv_results_ for grid searches for all 3 scoring options """
    scorers = ['accuracy', 'roc_auc', 'f1']
    # StratifiedKFold will ensure an equal distribution of the target classes
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    results = {}
    for scorer in scorers:
        grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=skf, scoring=scorer)
        grid_search.fit(X_train, y_train)
        results[scorer] = pd.DataFrame(grid_search.cv_results_)
    return results


if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    pipe = make_pipeline(
        StandardScaler(),
        SVC(random_state=0)
    )
    param_grid = {
        'svc__C': [0.1, 1]
    }
    results = validation_scores(pipe, param_grid, X_train, y_train)
    print('Validation results:')
    print(results)

    img = load_image(df.iloc[0]['image_name'], df.iloc[0]['class'])
    show_image(img)