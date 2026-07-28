from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from data_utils import load_data, get_target_label


df = load_data('archive/healthcare_dataset.csv')
y = get_target_label(df)
features = ['Age','Billing Amount','Gender','Blood Type','Admission Type','Medication','Test Results']
X = df[features].copy()
X['Gender'] = X['Gender'].astype(str).str.strip().str.title()
X['Blood Type'] = X['Blood Type'].astype(str).str.strip().str.upper()
X['Admission Type'] = X['Admission Type'].astype(str).str.strip().str.title()
X['Test Results'] = X['Test Results'].astype(str).str.strip().str.title()
num_features = ['Age','Billing Amount']
cat_features = ['Gender','Blood Type','Admission Type','Medication','Test Results']
pre = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_features),
], remainder='drop')
Xtr = pre.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(Xtr, y, test_size=0.2, stratify=y, random_state=42)

models = {
    'logreg_balanced': LogisticRegression(max_iter=4000, class_weight='balanced', solver='lbfgs'),
    'rf_balanced': RandomForestClassifier(n_estimators=250, class_weight='balanced_subsample', random_state=42, n_jobs=-1),
    'hgb': HistGradientBoostingClassifier(random_state=42),
    'gb': GradientBoostingClassifier(random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:,1]
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(y_test, preds)
        print(name, 'thr', threshold, 'acc', acc)
    print('\n', name)
    print(classification_report(y_test, (probs >= 0.3).astype(int), target_names=['No Cancer','Cancer']))
