import logging
import traceback
from flask import Flask, request
from utils import *
from queue import Queue
import threading
opset_version = 11
# 此处示例，需要根据模型类型重写
def init_model(model_path, export_model_path, optimized_model_path, length=32):
    model = torch.load(model_path).to(torch.device("cuda"))
    model.eval()

    if length == 32:
        data = [[[2, 16, 2874, 20, 3, 16, 36, 130, 5605, 458, 20, 3,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0]]]

    else:
        data = [[[2, 16, 2874, 20, 3, 16, 36, 130, 5605, 458, 2, 16, 2874, 20, 3, 16, 36, 130, 5605, 458, 2, 16, 2874, 20,
                  3, 16, 36, 130, 5605, 458, 2, 16, 2874, 20, 3, 16, 36, 130, 5605, 458, 2, 16, 2874, 20, 3, 16, 36, 130,
                  5605, 458, 2, 16, 2874, 20, 3, 16, 36, 130, 5605, 458, 2, 16, 2874, 20, 3, 16, 36, 130, 5605, 458, 2, 16,
                  2874, 20, 3, 16, 36, 130, 5605, 458, 2, 16, 2874, 20, 3, 16, 36, 130, 5605, 458, 2, 16, 2874, 20, 3, 16,
                  36, 130, 5605, 458]],
                [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
                  1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                  1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]


    inputs = {
        'input_ids': torch.tensor(data[0]).to(config.device),
        'input_masks': torch.tensor(data[1]).to(config.device),
        'segment_ids': torch.tensor(data[2]).to(config.device)
    }

    if True or not os.path.exists(export_model_path):
        with torch.no_grad():
            symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
            torch.onnx.export(model,  # model being run
                              args=tuple(inputs.values()),  # model input (or a tuple for multiple inputs)
                              f=export_model_path,  # where to save the model (can be a file or file-like object)
                              opset_version=opset_version,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['input_ids',  # the model's input names
                                           'input_masks',
                                           'segment_ids'],
                              output_names=['predict'],  # the model's output names
                              dynamic_axes={'input_ids': symbolic_names,  # variable length axes
                                            'input_masks': symbolic_names,
                                            'segment_ids': symbolic_names,
                                            'predict': symbolic_names})
            print("Model exported at ", export_model_path)

    from onnxruntime_tools import optimizer
    from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions
    opt_options = BertOptimizationOptions('bert')
    opt_options.enable_embed_layer_norm = False

    opt_model = optimizer.optimize_model(
        export_model_path,
        'bert',
        num_heads=12,
        hidden_size=768,
        optimization_options=opt_options)
    opt_model.save_model_to_file(optimized_model_path)

    del model
    torch.cuda.empty_cache()

    import psutil
    import onnxruntime

    assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()

    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
    session = onnxruntime.InferenceSession(optimized_model_path, sess_options)
    ort_inputs = {
        'input_ids': [[0]*32],
        'input_masks': [[0]*32],
        'segment_ids': [[0]*32]
        }
    session.run(None, ort_inputs)#预先启动一下
    return session

def infer(session,config,inp:Queue,res:Queue):
    data_gen = data_generator(config)
    while True:
        query_A, query_B=inp.get()#不断从自己队列中取
        input_ids, input_masks, segment_ids = data_gen.generate((query_A, query_B))
        ort_inputs = {
        'input_ids': input_ids,
        'input_masks': input_masks,
        'segment_ids': segment_ids
        }
        y_pred = session.run(None, ort_inputs)
        res.put(y_pred[0])#结果放入队列

def softmax(x, axis=1):
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max = row_max.reshape(-1, 1)
    x = x - row_max

    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


class Config:
    def __init__(self):
        # 预训练模型路径
        self.modelId = 2
        self.model = "NEZHA"
        self.Stratification = False

        self.model_path = 'model0/'
        self.num_class = 2
        self.dropout = 0.2
        self.MAX_LEN = 32
        self.epoch = 5
        self.learn_rate = 2e-5
        self.normal_lr = 1e-4
        self.batch_size = 1
        self.k_fold = 5
        self.seed = 42

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.focalloss = False
        self.pgd = False
        self.fgm = True

# 允许使用类似Flask的别的服务方式
app = Flask(__name__)

@app.route("/tccapi", methods=['GET', 'POST'])
def tccapi():
    data = request.get_data()
    if (data == b"exit"):
        print("received exit command, exit now")
        os._exit(0)
    input_list = request.form.getlist("input")
    index_list = request.form.getlist("index")

    response_batch = {}
    response_batch["results"] = []

    for i in range(len(index_list)):
        index_str = index_list[i]
        response = {}
        try:
            input_sample = input_list[i].strip()
            elems = input_sample.strip().split("\t")
            query_A = elems[0].strip()
            query_B = elems[1].strip()

            for i in runningModelIds[:3]:#先只用上前三个模型
                assert inps[i].qsize()==0
                inps[i].put((query_A,query_B))#为子线程提供数据
            predict_res=[]
            for i in runningModelIds[:3]:
                predict_res.append(res.get())#取3次
            assert res.qsize()==0
            y_pred = np.mean(predict_res, axis=0)
            y_pred = softmax(np.array(y_pred))
            if 0.15<float(y_pred[0][1])< 0.85:
                assert inps[3].qsize()==0
                inps[3].put((query_A,query_B))#为第四个子线程提供数据
                predict_res.append(res.get())
                assert res.qsize()==0
                y_pred = np.mean(predict_res, axis=0)
                y_pred = softmax(np.array(y_pred))


            response["predict"] = float(y_pred[0][1])
            response["index"] = index_str
            response["ok"] = True
        except Exception as e:
            response["predict"] = 0
            response["index"] = index_str
            response["ok"] = False
            traceback.print_exc()
        response_batch["results"].append(response)

    return response_batch



if __name__ == "__main__":
    # 此处示例，需要根据模型类型重写加载部分
    output_dir = "./onnx"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_lists = ["nezha-base-count3", "nezha-base-count5", "bert-base-count3", "bert-base-count5"]
    lens=[32,100,32,100]
    configs=[]
    sessions=[]
    for path,length in zip(model_lists,lens):
        config = Config()
        export_model_path = os.path.join(output_dir, 'opset{}.onnx'.format(path))
        optimized_model_path = os.path.join(output_dir, 'optimizer{}.onnx'.format(path))
        config.model_path = './{}/finetuning/models/'.format(path)
        config.MAX_LEN = length
        session = init_model(config.model_path+"bert_0.pth", export_model_path, optimized_model_path)
        sessions.append(session)
        configs.append(config)


    #多线程相关
    inps=[Queue() for i in range(4)]#存子线程输入
    res=Queue()#存结果
    proLs=[]
    runningModelIds=[0,1,2,3]#控制使用那几个模型，如果时间不够，可以在途中退出一个线程，除去一个模型

    for i in runningModelIds:
        proLs.append(threading.Thread(target=infer,args=(sessions[i],configs[i],inps[i],res)))
        proLs[-1].daemon = True#子线程随着主线程结束
        proLs[-1].start()#一次性全启动

    log = logging.getLogger('werkzeug')#关闭冗长的http 200 log
    log.disabled = True

    app.run(host="127.0.0.1", port=8080)

