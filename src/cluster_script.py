# cluster_data = pd.read_parquet('../data/song_cluster.parquet')

# cluster_data = cluster_data.rename(columns={'label': 'cluster'})
# session_group = train_data.groupby('session_id')
# subsample = session_group.get_group(751)
# merge_df = subsample.merge(cluster_data, on='song_id', how='left')
# unique_cluster = merge_df['cluster'].unique().tolist()
# elements = cluster_data[cluster_data['cluster'].isin(unique_cluster)]
# input_sample = np.random.choice(
#     elements['song_id'],
#     int(len(elements)*0.01)
# ).tolist()

"""
DCG script
==========
def dcg_score(y_true, y_pred, k=None):
    _, indices = torch.topk(y_pred, k, largest=True, sorted=True)
    rank = torch.arange(1, indices.shape[1] + 1, dtype=torch.float32)
    gains = 2 ** y_true[0, indices] - 1
    discounts = torch.log2(rank + 1)
    dcg = torch.sum(gains / discounts)
    return dcg

class DCGLoss(nn.Module):
    def __init__(self, k=None):
        super().__init__()
        self.k = k

    def forward(self, y_true, y_pred):
        return -dcg_score(y_true, y_pred, self.k)
"""