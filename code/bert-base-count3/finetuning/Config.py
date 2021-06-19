from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer, XLNetConfig, ElectraModel, ElectraConfig, ElectraTokenizer, \
    RobertaTokenizer, RobertaModel, RobertaConfig
from NEZHA.modeling_nezha import NeZhaModel
from NEZHA.configuration_nezha import NeZhaConfig


MODELS = {
    'BertForClass':  BertModel,
    'BertForClass_MultiDropout':  BertModel,
   'BertLastTwoCls':  BertModel,
    'BertLastCls':BertModel,
   'BertLastTwoClsPooler':  BertModel,
    'BertLastTwoEmbeddings': BertModel,
    'BertLastTwoEmbeddingsPooler': BertModel,
    'BertLastFourCls': BertModel,
    'BertLastFourClsPooler':  BertModel,
    'BertLastFourEmbeddings':  BertModel,
   'BertLastFourEmbeddingsPooler':  BertModel,
   'BertDynCls':  BertModel,
    'BertDynEmbeddings': BertModel,
    'BertRNN': BertModel,
    'BertCNN': XLNetModel,
    'BertRCNN':  BertModel,
    'XLNet': XLNetModel,
    'Electra': ElectraModel,
    'NEZHA': NeZhaModel
    }

TOKENIZERS = {
    'BertForClass': BertTokenizer,
    'BertForClass_MultiDropout': BertTokenizer,
    'BertLastTwoCls': BertTokenizer,
    'BertLastCls': BertTokenizer,
    'BertLastTwoClsPooler': BertTokenizer,
    'BertLastTwoEmbeddings': BertTokenizer,
    'BertLastTwoEmbeddingsPooler': BertTokenizer,
    'BertLastFourCls': BertTokenizer,
    'BertLastFourClsPooler': BertTokenizer,
    'BertLastFourEmbeddings': BertTokenizer,
    'BertLastFourEmbeddingsPooler': BertTokenizer,
    'BertDynCls': BertTokenizer,
    'BertDynEmbeddings': BertTokenizer,
    'BertRNN': BertTokenizer,
    'BertCNN': BertTokenizer,
    'BertRCNN': BertTokenizer,
    'XLNet': XLNetTokenizer,
    'Electra': ElectraTokenizer,
    'NEZHA': BertTokenizer
    }

CONFIGS = {
    'BertForClass': BertConfig,
    'BertForClass_MultiDropout': BertConfig,
    'BertLastTwoCls': BertConfig,
    'BertLastCls': BertConfig,
    'BertLastTwoClsPooler': BertConfig,
    'BertLastTwoEmbeddings': BertConfig,
    'BertLastTwoEmbeddingsPooler': BertConfig,
    'BertLastFourCls': BertConfig,
    'BertLastFourClsPooler': BertConfig,
    'BertLastFourEmbeddings': BertConfig,
    'BertLastFourEmbeddingsPooler': BertConfig,
    'BertDynCls': BertConfig,
    'BertDynEmbeddings': BertConfig,
    'BertRNN': BertConfig,
    'BertCNN': BertConfig,
    'BertRCNN': BertConfig,
    'XLNet': XLNetConfig,
    'Electra': ElectraConfig,
    'NEZHA': NeZhaConfig

    }