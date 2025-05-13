import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, accuracy_score, \
    precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

warnings.filterwarnings('ignore')


# 1. Veri Ön İşleme
def preprocess_data(df, le=None, fit=True):
    print(f"\nVeri ön işleme başladı (fit={fit})...")
    print(f"İşlem öncesi veri boyutu: {df.shape}")

    df.columns = df.columns.str.strip().str.upper()
    print(f"Sütun isimleri standartlaştırıldı: {df.columns.tolist()}")

    df = df.dropna(how='all')
    print(f"Boş satırlar temizlendikten sonra boyut: {df.shape}")

    label_col = df.columns[-1]
    print(f"Etiket sütunu belirlendi: {label_col}")

    if fit:
        le = LabelEncoder()
        df[label_col] = le.fit_transform(df[label_col])
        print(f"Etiketler kodlandı. Sınıflar: {le.classes_}")
    else:
        unique_labels = set(df[label_col].unique())
        known_labels = set(le.classes_)
        if not unique_labels.issubset(known_labels):
            print(f"Uyarı: Test verisinde eğitimde görülmeyen etiketler var: {unique_labels - known_labels}")
            df = df[df[label_col].isin(known_labels)]  # Remove unseen labels
        df[label_col] = le.transform(df[label_col])
        print(f"Etiketler mevcut encoder ile dönüştürüldü.")

    # Aykırı değer temizleme
    if fit:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(label_col, errors='ignore')
        if not numeric_cols.empty:
            print(f"\nAykırı değer analizi başlıyor...")
            print(f"Nümerik sütunlar: {numeric_cols.tolist()}")

            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1

            mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
                     (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

            removed_outliers = len(df) - sum(mask)
            df = df[mask]
            print(f"Toplam {removed_outliers} aykırı değer kaldırıldı.")
            print(f"Aykırı değer temizlendikten sonra boyut: {df.shape}")
        else:
            print("Aykırı değer analizi için nümerik sütun bulunamadı.")

    print(f"\nVeri ön işleme tamamlandı. Son veri boyutu: {df.shape}")
    return df, le, label_col


# 2. Normalizasyon ve Özellik Seçimi Fonksiyonu
def normalize_and_select_features(X_train, X_test, y_train, method='pca'):
    print(f"\n{method.upper()} yöntemiyle özellik seçimi başlıyor...")

    print("\nStandardScaler ile normalizasyon yapılıyor...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Normalizasyon tamamlandı. Eğitim verisi ortalaması: {np.mean(X_train_scaled, axis=0)[:5]}...")
    print(f"Normalizasyon tamamlandı. Test verisi ortalaması: {np.mean(X_test_scaled, axis=0)[:5]}...")

    # Özellik seçimi
    if method == 'pca':
        n_components = min(5, X_train.shape[1])
        print(f"\nPCA uygulanıyor. n_components={n_components}")
        selector = PCA(n_components=n_components)
    elif method == 'lda':
        n_components = min(5, len(np.unique(y_train)) - 1, X_train.shape[1])
        print(f"\nLDA uygulanıyor. n_components={n_components}")
        selector = LDA(n_components=n_components)
    elif method == 'rfe':
        n_features = min(5, X_train.shape[1])
        print(f"\nRFE uygulanıyor. n_features_to_select={n_features}")
        selector = RFE(LogisticRegression(max_iter=1000), n_features_to_select=n_features)
        # Fix for RFE: Add y_train parameter
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        print(f"Seçilen özellikler: {selector.support_}")
        print(f"Özellik boyutu: {X_train.shape[1]} -> {X_train_selected.shape[1]}")
        return X_train_selected, X_test_selected, selector, scaler
    else:
        print("\nÖzellik seçimi yapılmadı.")
        return X_train_scaled, X_test_scaled, None, None

    print("\nFeature selector eğitiliyor...")
    if method == 'lda':
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    else:
        X_train_selected = selector.fit_transform(X_train_scaled)

    X_test_selected = selector.transform(X_test_scaled)

    if method == 'pca':
        print(f"PCA bileşen varyans oranları: {selector.explained_variance_ratio_}")
    elif method == 'lda':
        print(f"LDA sınıf ayrımı başarısı: {selector.explained_variance_ratio_}")
    elif method == 'rfe':
        print(f"Seçilen özellikler: {selector.support_}")

    print(f"Özellik boyutu: {X_train.shape[1]} -> {X_train_selected.shape[1]}")

    return X_train_selected, X_test_selected, selector, scaler


def main():
    results_dir = 'results'
    subdirs = {
        'graphs': ['confusion_matrices', 'metric_comparisons'],
        'reports': ['classification_reports', 'metric_values'],
        'models': ['saved_models'],
        'feature_selectors': ['pca', 'lda', 'rfe']
    }

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"'{results_dir}' klasörü oluşturuldu.")
    else:
        print(f"'{results_dir}' klasörü zaten var.")

    for category, folders in subdirs.items():
        category_path = os.path.join(results_dir, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

        for folder in folders:
            folder_path = os.path.join(category_path, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"'{folder_path}' alt klasörü oluşturuldu.")

    print("\n" + "=" * 50)
    print("VERİ YÜKLEME VE ÖN İŞLEME")
    print("=" * 50)

    try:
        print("\nEĞİTİM VERİSİ İŞLENİYOR...")
        train_df = pd.read_excel('veri_seti/traindata.xlsx')
        print("Eğitim verisi yüklendi. Boyut:", train_df.shape)
        train_df, le, label_col = preprocess_data(train_df, fit=True)

        print("\nTEST VERİSİ İŞLENİYOR...")
        test_df = pd.read_excel('veri_seti/testdata.xlsx')
        print("Test verisi yüklendi. Boyut:", test_df.shape)
        test_df, _, _ = preprocess_data(test_df, le=le, fit=False)

    except Exception as e:
        print(f"\n!!! VERİ YÜKLEME HATASI: {str(e)}")
        return

    print("\n" + "=" * 50)
    print("NORMALİZASYON VE ÖZELLİK SEÇİMİ")
    print("=" * 50)

    print("\nVeri eğitim ve test setlerine ayrılıyor...")
    X_train = train_df.drop(label_col, axis=1)
    y_train = train_df[label_col]
    X_test = test_df.drop(label_col, axis=1)
    y_test = test_df[label_col]
    print(f"Eğitim verisi boyutu: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test verisi boyutu: X_test={X_test.shape}, y_test={y_test.shape}")

    feature_selectors = {}
    for method in ['pca', 'lda', 'rfe']:
        try:
            print("\n" + "-" * 50)
            print(f"{method.upper()} YÖNTEMİ UYGULANIYOR")
            print("-" * 50)
            X_train_sel, X_test_sel, selector, scaler = normalize_and_select_features(X_train, X_test, y_train, method)
            feature_selectors[method] = (X_train_sel, X_test_sel, selector, scaler)

            # Feature selector'ı kaydet
            if selector is not None:
                selector_path = os.path.join(
                    results_dir, 'feature_selectors', method,
                    f'{method}_selector.pkl'
                )
                joblib.dump(selector, selector_path)
                print(f"{method.upper()} selector kaydedildi: {selector_path}")

                # Scaler'ı kaydet
                scaler_path = os.path.join(
                    results_dir, 'feature_selectors', method,
                    f'{method}_scaler.pkl'
                )
                joblib.dump(scaler, scaler_path)
                print(f"{method.upper()} scaler kaydedildi: {scaler_path}")

        except Exception as e:
            print(f"\n!!! {method} HATASI: {str(e)}")

    print("\n" + "=" * 50)
    print("MODEL EĞİTİMİ VE DEĞERLENDİRME")
    print("=" * 50)

    models = {
        'SVM': SVC(kernel='rbf', probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'Linear Regression': LinearRegression(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    }

    results = {}
    for method, (X_train_sel, X_test_sel, selector, _) in feature_selectors.items():
        for name, model in models.items():
            try:
                print("\n" + "-" * 50)
                print(f"{method.upper()} - {name} MODELİ EĞİTİLİYOR...")
                print("-" * 50)

                # Model eğitimi
                print("Model eğitimi başladı...")
                model.fit(X_train_sel, y_train)
                print("Model eğitimi tamamlandı.")

                # Tahminler
                if name == 'Linear Regression':
                    print("Lineer regresyon için tahminler yuvarlanıyor...")
                    y_pred = np.clip(np.round(model.predict(X_test_sel)), 0, len(le.classes_) - 1)
                else:
                    y_pred = model.predict(X_test_sel)

                # Metrikler
                print("\nModel performans metrikleri hesaplanıyor...")
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                mse = mean_squared_error(y_test, y_pred)

                results[f"{method}_{name}"] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'mse': mse,
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'model': model
                }

                # Karışıklık matrisi
                plt.figure(figsize=(8, 6))
                sns.heatmap(results[f"{method}_{name}"]['confusion_matrix'],
                            annot=True, fmt='d', cmap='Blues',
                            xticklabels=le.classes_, yticklabels=le.classes_)
                plt.title(f"{method.upper()} - {name}")
                confusion_matrix_path = os.path.join(
                    results_dir, 'graphs', 'confusion_matrices',
                    f"{method}_{name}_confusion_matrix.png"
                )
                plt.savefig(confusion_matrix_path)
                plt.close()
                print(f"Karışıklık matrisi kaydedildi: {confusion_matrix_path}")

                # Sınıflandırma raporu
                report = classification_report(y_test, y_pred,
                                               target_names=le.classes_,
                                               labels=np.unique(y_test),
                                               output_dict=True,
                                               zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                report_path = os.path.join(
                    results_dir, 'reports', 'classification_reports',
                    f"{method}_{name}_classification_report.csv"
                )
                report_df.to_csv(report_path)
                print(f"Sınıflandırma raporu kaydedildi: {report_path}")

            except Exception as e:
                print(f"\n!!! {method}_{name} HATASI: {str(e)}")

    # Performans
    print("\n" + "=" * 50)
    print("PERFORMANS DEĞERLENDİRMESİ")
    print("=" * 50)

    if results:
        # Metrikleri görselleştirme
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'mse']
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            plt.bar(results.keys(), [r[metric] for r in results.values()])
            plt.xticks(rotation=45)
            plt.title(metric.upper())
            plt.ylabel(metric)
            plt.tight_layout()

            metric_path = os.path.join(
                results_dir, 'graphs', 'metric_comparisons',
                f"{metric}_comparison.png"
            )
            plt.savefig(metric_path)
            plt.close()
            print(f"{metric.upper()} karşılaştırma grafiği kaydedildi: {metric_path}")

            # Perfomans Metiklerini csv olarak kaydet
            metric_values = {model_name: res[metric] for model_name, res in results.items()}
            metric_df = pd.DataFrame.from_dict(metric_values, orient='index', columns=[metric])
            metric_csv_path = os.path.join(
                results_dir, 'reports', 'metric_values',
                f"{metric}_values.csv"
            )
            metric_df.to_csv(metric_csv_path)
            print(f"{metric.upper()} değerleri kaydedildi: {metric_csv_path}")

        # Tüm sonuçlar
        all_results = []
        for model_name, res in results.items():
            row = {'Model': model_name}
            row.update({k: v for k, v in res.items() if k != 'confusion_matrix' and k != 'model'})
            all_results.append(row)

        results_df = pd.DataFrame(all_results)
        results_csv_path = os.path.join(
            results_dir, 'reports',
            'all_results.csv'
        )
        results_df.to_csv(results_csv_path, index=False)
        print(f"Tüm sonuçlar kaydedildi: {results_csv_path}")

        # Mettiklere göre en iyi modeli kaydet
        best_model_name = max(results, key=lambda k: results[k]['accuracy'])
        best_model = results[best_model_name]['model']

        best_model_path = os.path.join(
            results_dir, 'models',
            'best_model_info.txt'
        )
        with open(best_model_path, 'w') as f:
            f.write(f"En iyi model: {best_model_name}\n")
            f.write(f"Accuracy: {results[best_model_name]['accuracy']:.4f}\n")
            f.write(f"Precision: {results[best_model_name]['precision']:.4f}\n")
            f.write(f"Recall: {results[best_model_name]['recall']:.4f}\n")
            f.write(f"F1 Score: {results[best_model_name]['f1']:.4f}\n")
            f.write(f"MSE: {results[best_model_name]['mse']:.4f}\n")

        print(f"En iyi model bilgileri kaydedildi: {best_model_path}")

        joblib.dump(best_model, os.path.join(
            results_dir, 'models', 'saved_models',
            'best_model.pkl'
        ))
        joblib.dump(le, os.path.join(
            results_dir, 'models', 'saved_models',
            'label_encoder.pkl'
        ))
        print("En iyi model ve label encoder 'results' klasörüne kaydedildi.")
    else:
        print("\n!!! HİÇBİR MODEL BAŞARIYLA ÇALIŞTIRILAMADI !!!")

    print("\nTÜM İŞLEMLER TAMAMLANDI! SONUÇLAR 'results' KLASÖRÜNDE KAYDEDİLDİ.")


if __name__ == "__main__":
    main()