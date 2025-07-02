import os
import numpy as np
import wfdb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import joblib

def load_and_segment_ecg_data(data_dir, segment_length=360):
    segments = []
    labels = []

    # Only use records that have an .atr annotation file
    record_names = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.atr')]

    for record_name in tqdm(record_names):
        try:
            # Load signal and annotations
            record_path = os.path.join(data_dir, record_name)
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')

            signal = record.p_signal[:, 0]  # Use only MLII or the first channel

            for sample, symbol in zip(annotation.sample, annotation.symbol):
                # Only keep beat classes of interest
                if symbol in ['N', 'L', 'R', 'A', 'V', 'F']:
                    start = sample - segment_length // 2
                    end = sample + segment_length // 2

                    # Ensure window is valid
                    if start >= 0 and end <= len(signal):
                        segment = signal[start:end]
                        if len(segment) == segment_length:
                            segments.append(segment)
                            labels.append(symbol)

        except Exception as e:
            print(f"Error processing {record_name}: {e}")

    if len(segments) == 0:
        raise ValueError("No segments extracted. Check dataset format and annotations.")

    segments = np.array(segments)
    labels = np.array(labels)

    print(f"Extracted {len(segments)} segments.")

    # Normalize the segments
    scaler = StandardScaler()
    segments = scaler.fit_transform(segments)

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Save everything
    np.save(os.path.join(data_dir, 'ecg_segments.npy'), segments)
    np.save(os.path.join(data_dir, 'ecg_labels.npy'), labels_encoded)
    joblib.dump(label_encoder, os.path.join(data_dir, 'label_encoder.pkl'))

    return segments, labels_encoded
