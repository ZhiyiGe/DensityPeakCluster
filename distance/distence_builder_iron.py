import pandas as pd
from DensityPeakCluster.distance.distance import *
from DensityPeakCluster.distance.distance_builder import DistanceBuilder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt



class Builder(DistanceBuilder):
    def __init__(self,ncomponents , neighbor):
        super().__init__()
        self.neighbor = neighbor
        self.ncomponents = ncomponents

    def load_points(self, filename):
        data = pd.read_excel(filename).drop('时间戳', axis=1)  # 读取数据
        result = data['聚类结果']
        vecoters = data.drop('聚类结果', axis=1).values
        self.vectors = vecoters
        self.result = result
        self.data_preprocess()

    def data_preprocess(self):
        processer = data_process(self.vectors)
        processer.PCA_dim_reduce(self.ncomponents)
        # processer.Isomap_dim_reduce(self.ncomponents,self.neighbor)
        processer.normalize()
        self.vectors = processer.data
        # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # for idx, neighbor in enumerate([i for i in range(2,200)]):
        #     processer.Isomap_dim_reduce(2,288)
        #     new_X_isomap=processer.data
        #     plt.scatter(new_X_isomap[:, 0], new_X_isomap[:, 1], c=self.result)
        #     plt.title("Isomap (n_neighbors=%d)" % neighbor)
        #     plt.show()

    def build_result_file_for_cluster(self, filename):
        """
        Save result and index into file

        Args:
            filename     : file to save the result for cluster
        """
        fo = open(filename, 'w')
        fo.write('\n'.join(map(str, self.result)))
        fo.close()


class data_process:
    def __init__(self, data):
        self.data = data

    def normalize(self):
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data.T).T

    def PCA_dim_reduce(self, ncomponents):
        pca = PCA(n_components=ncomponents)
        self.data = pca.fit_transform(self.data)

    def LDA_dim_reduce(self, ncomponents):
        lda = LDA(n_components=ncomponents)
        self.data = lda.fit_transform(self.data)

    def MDS_dim_reduce(self, ncomponents):
        mds = MDS(n_components=ncomponents, metric=True)
        self.data = mds.fit_transform(self.data)

    def Isomap_dim_reduce(self, ncomponents, neighbor):
        isomap = Isomap(n_components=ncomponents, n_neighbors=neighbor)
        self.data = isomap.fit_transform(self.data)


if __name__ == '__main__':
    builder = Builder(2,None)
    builder.load_points(r'../data/data_iron/data.xlsx')
    builder.build_result_file_for_cluster(r'../data/data_iron/result.txt')
    # # 计算余弦距离变更保存
    builder.build_distance_file_for_cluster(ConsineDistance(), r'../data/data_iron/distence.dat')
