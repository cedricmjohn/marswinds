import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
from marswinds.utils import radian_to_degree
from marswinds.predict.predict import Predictor
from tqdm import tqdm

class TestModelPredictions(Predictor):
    def __init__(self,probability_threshold = 0.97, planet='EARTH'):
        super().__init__(probability_threshold, planet)
        self.initialize_results()

    def get_label(self,image_name):
        parts = image_name[:-4].split('_')
        sin = float(parts[-3])
        cos = float(parts[-2])
        angle = radian_to_degree(math.atan2(sin, cos))
        if angle<0: angle+=360
        wind_strength = float(parts[-1])
    
        return(wind_strength, angle)

    def calculate_angular_diff(self, label_angle, pred_angle):
        if label_angle <0:
            label_angle = 360+label_angle
        if pred_angle <0:
            pred_angle = 360+pred_angle
    
        angle_diff = np.abs(label_angle-pred_angle)
        if angle_diff > 180: angle_diff = 360-angle_diff
        
        return angle_diff
    
    def compute_RMSE(diff):
        return np.sqrt(np.mean(diff**2))
    
    def compute_classification_metrics(df):
        
        tp = df[(df.is_dune == df.dune_label) & (df.is_dune_label == 1)].shape[0]
        tn = df[(df.is_dune == df.dune_label) & (df.is_dune_label == 0)].shape[0]
        fp = df[(df.is_dune != df.dune_label) & (df.is_dune_label == 1)].shape[0]
        fn = df[(df.is_dune != df.dune_label) & (df.is_dune_label == 0)].shape[0]
        
        precision = tp/(tp+fp)
        recall = (tp+tn)/(tp+fp+tn+fp)
        
        return (precision, recall)
        

    def compute_metrics(self, file_path):
        results = pd.read_csv(file_path)
        results['angular_diff'] = []
        
        for index, result in results.iterrows():
            if type(result.angle) == float:
                diff = self.calculate_angular_diff(result.angle_label, result.angle)
                results.loc[index, 'angular_diff'] = diff
            
        precision, recall = self.compute_classification_metrics(results)
        rmse = self.compute_RMSE(results.angular_diff)
    
    
        return (precision, recall, rmse)


    def plot_rand_image_grid(self,grid_dim=5,image_dim=3, 
                             selected_results=None, full_path=False, 
                             folder_name='testing', 
                             image_type='dunes'):
        fig, axes = plt.subplots(grid_dim,grid_dim,figsize=(grid_dim*image_dim,grid_dim*image_dim))
        for ax in axes.flatten():
            self.plot_image(ax=ax, selected_results=selected_results, full_path = full_path, folder_name=folder_name, image_type=image_type)
        fig.tight_layout()
        return fig
        

    def convert_label(self,label):
        if label > 0:
            label = 'dunes'
        else:
            label = 'no_dunes'
        return label
    
    
    def test_from_file(self, path, 
                          pixel_resolution=10,
                          labels = None,
                          coordinates=None,
                          label_wind=True):
        
        results, prediction_folder_path,_ =  self.get_predictions(path, pixel_resolution, coordinates)
    
        wind_strength_label = np.nan
        angle_label = np.nan
        is_dune_label = np.nan
        
        if labels == None:
            filename = path.split('/')[-1]
            wind_strength_label, angle_label = self.get_label(filename)
            if path.split('/')[-2] == 'dunes':
                is_dune_label = True
            if path.split('/')[-2] == 'no_dunes':
                is_dune_label = False
            results['strength_label'] = [wind_strength_label]
            results['angle_label'] = [angle_label]
            results['is_dune_label'] = [is_dune_label]
        else:
            labels_df = pd.read_csv(labels)
            results.to_csv(f'{prediction_folder_path}/predictions_premerge.csv', index=False)
            results = pd.merge(results,labels_df[['filename','strength_label',
                                    'angle_label','is_dune_label']], 
                                    on='filename',how='left')
        
        
        
        results.to_csv(f'{prediction_folder_path}/predictions.csv', index=False)
        
        return results
    
    def predict_from_folders(self, foldernames, pixel_resolution=10):
        return self.results
    
    def test_single_image(self, path, labels, 
                          pixel_resolution=10, 
                          predict_class=True, 
                          predict_direction=True):
        self.predict_from_file(path,pixel_resolution)
        
        if labels == 'from_filename':
            pass
        if labels == 'from_csv':
            pass
        return None
        
    def tile_and_label(self, path_to_label_files):

        images = pd.read_csv(path_to_label_files)

        for index, row in images.iterrows():
            image_path = row['filename']
            image=cv2.imread(image_path)
            plt.imshow(image)
            plt.show()
            answer = input('contains dunes? (y)es | (n)o ')
            if answer == 'y':
                answer = 1
            else:
                answer = 0
                
            images.loc[index,'is_dune_label'] = answer
        print(images)
        images.to_csv(path_to_label_files)
            
if __name__ == '__main__':
    print('problem')
    #image = '../raw_data/images/testing/dunes/21.8424991607666_50.2474983215332_031_CW000_-0.6031867265701294_-0.6447361707687378_1.9967776536941528.jpg'
    image = '../raw_data/earth_images/landsat-106.473427_32.923187_-106.27342700000001_33.123187.png'
    labels = '../raw_data/earth_images/predictions_landsat-106.473427_32.923187_-106.27342700000001_33.123187.png/label_data.csv'
    pred = '../raw_data/earth_images/predictions_landsat-106.473427_32.923187_-106.27342700000001_33.123187.png/predictions.csv'
    #image = '../raw_data/mars_images/alt/Mars_small_new.jpg'
    #path,df = predictor.get_prediction_image(image,10)
    #predictor.predict_from_file('../raw_data/earth_images/landsat-106.473427_32.923187_-106.27342700000001_33.123187.png')
    tester = TestModelPredictions(probability_threshold=0.5)
    tester.save_tiled_data(image, coordinates=(32.7982115,-106.3750148),label_wind=False)
    tester.test_from_file(image, labels=labels, coordinates=(32.7982115,-106.3750148))
    #tester.tile_and_label(pred)

