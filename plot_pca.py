import os
import pickle as pk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import OmegaConf
from sklearn.base import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def plot_PCA(args):
    """
    function for executing PCA and showing plot
    """
    feature_type = args.get('feature_type')
    split_type = args.get('split_type')
    features_path = args.get('features_path', f'./saved_features/{feature_type}_{split_type}.pkl')
    samples_path = args.get('samples_path', f'./train_val/{split_type}.pkl')
    base_image_path = args.get('base_image_path', f'../ek_data/frames/')
    output_image_path = args.get('output_image_path', f'./plots')
    plot_3d = args.get('plot_3d', False)

    features_data, split_data = load_data(features_path, samples_path)
    central_frames = extract_central_frames(split_data, features_data, base_image_path)
    
    # applying PCA to reduced features
    reduced_features = apply_pca(features_data['features'], plot_3d)
    label_actions = defaultdict(set)
    for idx in range(len(split_data)):
        label_actions[split_data['verb_class'][idx]].add(split_data['verb'][idx])

    for label, acts in label_actions.items():
        label_actions[label] = ', '.join(acts)
    
        
    cluster_and_plot(reduced_features, central_frames, split_data, label_actions, output_image_path, plot_3d)

def load_data(features_path, samples_path):
    """
    loading data from features and samples
    """
    with open(features_path, 'rb') as feat_file:
        features_data = pk.load(feat_file)

        
    with open(samples_path, 'rb') as sample_file:
        split_data = pk.load(sample_file)
    
    return features_data, split_data

def extract_central_frames(samples, features_data, base_image_path):
    """
    extracting central frames from images
    """
    central_frames = []
    sample_central_frames = samples['start_frame'] + (samples['stop_frame'] - samples['start_frame']) // 2
    video_names = [x['video_name'] for x in features_data['features']]
    
    for idx in range(len(video_names)):
        central_frames.append(os.path.join(base_image_path, f"{video_names[idx]}/img_{sample_central_frames[idx]:010d}.jpg"))
    
    return central_frames

def apply_pca(features, plot_3d=False):
    """
    PCA analysis
    """
    pca = PCA(3 if plot_3d else 2)
    reduced_features = np.mean([x['features_RGB'] for x in features], 1)
    reduced_features = pca.fit_transform(reduced_features)
    
    # if not 3d third dimension = 0
    if not plot_3d:
        reduced_features = np.hstack((reduced_features, np.zeros((reduced_features.shape[0], 1))))
    
    return reduced_features

def cluster_and_plot(reduced_features, central_frames, split_data, label_actions, output_image_path, plot_3d=False):
    """
    clustering using K-means and plotting results
    """
    km = KMeans(n_clusters=8, random_state=8)
    km.fit(reduced_features)
    km.predict(reduced_features)
    centroids = km.cluster_centers_
    c_labels = km.labels_

    # cluster label for centroids
    centroid_labels = []
    for centroid in centroids:
        distances = np.linalg.norm(reduced_features - centroid, axis=1)
        closest_index = np.argmin(distances)
        centroid_label = c_labels[closest_index]
        centroid_labels.append(centroid_label)

    # extracting labels
    labels = split_data['verb_class'] #id - numerical label
     
    actions = [label_actions[label] for label in labels] # list of actions

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d') if plot_3d else fig.add_subplot()
    print(centroid_labels)
    label_colors = {}
    unique_actions = set(actions)
    colors = matplotlib.cm.get_cmap('rainbow_r', len(unique_actions))
    for i, action in enumerate(unique_actions):
        label_colors[action] = colors(i)

    # plotting scatter
    for i, (x, y, z) in enumerate(reduced_features):
        action = actions[i]
        ax.scatter(x, y, c=label_colors[action], label=action, s=100, zorder=1)
        ax.plot([x], [y], [z], marker='o', markersize=5, color=label_colors[action], alpha=1, zorder=1)

    # plotting centroids
    for i, centroid in enumerate(centroids):
        action = label_actions[i]
        print(action)
        print(centroid)
        ax.scatter(centroid[0], centroid[1], color=label_colors[action], marker='^', s=300, edgecolors='black', linewidths=2, alpha=1, zorder=3)
        ax.plot([centroid[0]], [centroid[1]], color=label_colors[action], markersize=10, zorder=3)
        ax.text(centroid[0], centroid[1], action, color='black', fontsize=12, zorder=3) #text to be deleted

    # legend colors
    legend_patches = [matplotlib.patches.Patch(color=color, label=label) for label, color in label_colors.items()]
    ax.legend(handles=legend_patches)

    plt.savefig(output_image_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    args = OmegaConf.from_cli()
    print(args)

    feature_type = args.feature_type
    split_type = args.split_type
    plot_3d = args.plot_3d

    plot_PCA(args)
