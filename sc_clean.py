import cv2
import pytesseract
import os
import shutil
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


input_folder = # insert filepath of the images 
output_folder = # insert filepath where output needs to be saved


num_clusters = 6 
ui_stop_words = [ 'file', 'edit', 'view', 'history', 'bookmarks', 'tools', 'help', 
'search', 'google', 'chrome', 'window', 'tab', 'menu', 'settings'
, 'battery', 'wifi', 'pm', 'am', 'type', 'here', 'dashboard', 'home',
'screen', 'shot', 'screenshot']

print(f"Copy & Arrange from {input_folder} ---")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_data = []  
documents = []  
low_text_files = [] 

#Main1
total_files = len([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"Found {total_files} images.")

counter = 0
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        full_path = os.path.join(input_folder, filename)
        counter += 1
        
        try:
            #step1
            img = cv2.imread(full_path)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            #step2
            text = pytesseract.image_to_string(gray)
            
            #step3
            clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
            valid_words = [w for w in clean_text.split() if len(w) > 2]
            final_text = " ".join(valid_words)
            if len(valid_words) < 5:
                low_text_files.append(filename)
            else:
                documents.append(final_text)
                file_data.append(filename)
            if counter % 10 == 0:
                print(f"Processed {counter}/{total_files}")
        except Exception as e:
            print(f"Error on {filename}: {e}")


#Main2
print(f"Clustering {len(documents)} images ")

if len(documents) < num_clusters:
    print("Not enough images to cluster")
    exit()

vectorizer = TfidfVectorizer(
    stop_words='english', 
    max_df=0.8,              
    min_df=2,                
)
my_stop_words = list(vectorizer.get_stop_words()) + ui_stop_words
vectorizer.stop_words = my_stop_words

X = vectorizer.fit_transform(documents)
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(X)

#Main 3
print(f"Copying files to {output_folder} ---")

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(num_clusters):
    top_terms = [terms[ind] for ind in order_centroids[i, :3]]
    folder_name = f"Group_{i}_" + "_".join(top_terms)
    
    #step1
    target_folder = os.path.join(output_folder, folder_name)
    os.makedirs(target_folder, exist_ok=True)
    
    print(f"Created {folder_name}")

    for index, label in enumerate(kmeans.labels_):
        if label == i:
            filename = file_data[index]
            src = os.path.join(input_folder, filename)
            dst = os.path.join(target_folder, filename)
            shutil.copy2(src, dst) 

#Main4
if low_text_files:
    review_folder = os.path.join(output_folder, "review_low_text")
    os.makedirs(review_folder, exist_ok=True)
    print(f"Copying {len(low_text_files)} unreadable images to 'review_low_text'")
    
    for filename in low_text_files:
        src = os.path.join(input_folder, filename)
        dst = os.path.join(review_folder, filename)
        shutil.copy2(src, dst)

print("\n No changes in the original folder")
print(f"Check {output_folder} for your desired output")