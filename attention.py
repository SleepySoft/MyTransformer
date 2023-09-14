import math
import torch


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    # 将K转置
    KT = K.permute(0, 1, 3, 2)
    # Q乘KT
    score = torch.matmul(Q, KT)
    # 根号dk
    d = K.shape[-1]
    d_sqrt = d ** 0.5
    # 除以根号dk
    score /= d_sqrt
    # mask
    score_masked = score.masked_fill(mask, -float('inf'))
    # softmax
    score_softmax = torch.softmax(score_masked, -1)
    # softmax之后乘以V
    result = torch.matmul(score_softmax, V)

    return result


class AttentionHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass


# 此处使用torch.nn.Embedding实现embedding层，它提供了一个简单的查找表，用于存储固定字典和大小的嵌入
# 注意该模块的权重是随机初始化的，这种做法通常被称为嵌入学习（Embedding Learning）或表示学习（Representation Learning）
# 与之对应的做法是使用预训练的词嵌入（Pre-trained Word Embeddings）
# 这里有两个问题：词向量之间的相关性（狗和猫）和上下文的相关性（saw a saw）。
# 对于词向量之间的相关性，嵌入学习也能够在一定程度上学习到词之间的关联，但表现肯定不如预训练的词嵌入
# 而对于上下文的相关性，传统的词嵌入方法（如Word2Vec或GloVe）是无法解决的。
# 关于上下文相关的词嵌入（Contextual Word Embeddings），最著名的就是BERT模型。在这种方法中，一个词的embedding会根据它所在的上下文进行调整。
# 当然使用torch.nn.Embedding是不可能做到上下文相关的,因此在本程序中embedding是固定编码并且随着模型一起训练的。

class TextEmbedding(torch.nn.Module):
    def __init__(self, token_list: [], embedding_dim: int):
        super(TextEmbedding, self).__init__()

        # 从外界传入的词表。注意torch.nn.Embedding只是根据index取得对应的embedding vector，所以我们需要建立从词到index的映射。
        self.token_list = token_list
        self.token_table = { token : i for i, token in enumerate(token_list) }
        # 定义一个embedding层。num_embeddings为词表的大小，embedding_dim为单个词编码的长度。
        self.embedding = torch.nn.Embedding(num_embeddings=len(token_list), embedding_dim=embedding_dim)
        # 初始化嵌入层的权重。.normal_(0, 0.1)表示从均值为0，标准差为0.1的正态分布中抽取样本来初始化权重。
        self.embedding.weight.data.normal_(0, 0.1)

    def forward(self, x: torch.Tensor):
        return self.embedding(x)


class PositionEmbedding(torch.nn.Module):
    def __init__(self, embedding_shape: (int, int)):
        """
        :param embedding_shape: (sequence size, embedding size)
        """
        super(PositionEmbedding, self).__init__()

        # 创建一个预计算好的position embedding表并将其注册为pe，接下来我们可以用self.pe访问这个表。
        self.create_pe_buffer(embedding_shape)

    def create_pe_buffer(self, embedding_shape: (int, int)):
        # Create a empty tensor by the size of embedding (the length of input sequence, the embedding length)
        pe = torch.empty(*embedding_shape)
        for pos in self.pe.shape[0]:
            for i in self.pe.shape[1]:
                self.pe[pos, i] = PositionEmbedding.calc_pe(pos, i, self.pe.shape[0])
        # Add an new dimension on dimension 0
        pe = pe.unsqueeze(0)
        # Register as a constant buffer
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # Add position embedding to text embedding
        # Note that the 0 dimension of input x is the batch size
        # pe will broadcast the dimension 0 to the same batch size of x and add
        return x + self.pe

    @staticmethod
    def calc_pe(pos: int, i: int, d_model: int):
        # 和论文中的实现一致，请对照阅读
        return math.sin(pos / 10000 ** (i / d_model)) if i % 2 == 0 else math.cos(pos / 10000 ** (i / d_model))


# Token: '0' - '9', 起始标记: '+', 终止标记: '-', Padding: '*'
TOKEN_LIST = ['+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '*']
EMBEDDING_LEN = 16
SEQUENCE_LEN = 32


class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.self_attention = None
        self.full_connection = None

    def add_location_encoder(self):
        pass


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_embedding = TextEmbedding(TOKEN_LIST, EMBEDDING_LEN)
        self.position_embedding = PositionEmbedding((SEQUENCE_LEN, EMBEDDING_LEN))

        self.encoder = EncoderLayer()

    def forward(self, x: torch.Tensor):
        pass
