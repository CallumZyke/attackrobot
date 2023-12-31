import gc

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from AttackFromDemo.llm_attacks import get_embedding_matrix, get_embeddings



def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    #embed_weights =
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    '''input_embeds = (one_hot @ embed_weights):
#这里是先假设one_hot形状为(batch_size, vocab_size)

    这是一个矩阵相乘的操作，其中
    one_hot
    是一个形状为(batch_size, vocab_size)
    的
    one - hot
    编码的张量，而
    embed_weights
    是一个形状为(vocab_size, embedding_dim)
    的嵌入权重矩阵。
    结果是一个形状为(batch_size, embedding_dim)
    的张量，其中包含了输入切片中每个
    token
    的嵌入表示。'''

    
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:],  #表示在整个输入序列中，选择从开头到 input_slice.start 之前的嵌入表示。
            input_embeds, 
            embeds[:,input_slice.stop:,:]   #目的是获取整个输入序列中，除了输入切片之外的部分的嵌入表示。
        ], 
        dim=1)

    # .logits:从模型的输出中获取对数概率
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
    
    loss.backward()
    
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    
    return grad



#后期有可能会变化
policy_device='cuda:0'

#未测试
#policy_device直接作为输入
def token_gradients_VIMA(policy_device, input_ids, actions,prefix_slice)
 #                        input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.

    Parameters
    ----------

    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.


    Args:
        actions:
            (从vima中获得的字典,注意不是env.action_space)
            {
                "pose0_position": [ , ], "pose0_rotation": [ , , , ],
                "pose1_position": [ , ], "pose1_rotation": [ , , , ]
            }



    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix_t5(model_name="t5-base")
    # chatGPT:创建一个全零的tensor，形状为 (batch_size, vocab_size)，其中 batch_size 是输入的样本数量，vocab_size 是嵌入矩阵的大小
    one_hot = torch.zeros(
        #input_ids[input_slice].shape[0],  # batch_size
      #  input_ids.shape[0] #
        input_ids[prefix_slice].shape[0],  # batch_size
        embed_weights.shape[0],  # vocab_size
        #device=model.device,
        device=policy_device,
        dtype=embed_weights.dtype
    )
    #未测试
    one_hot.scatter_(
        1,
        #input_ids[input_slice].unsqueeze(1),
       # input_ids.unsqueeze(1),
        input_ids[prefix_slice].unsqueeze(1),
        #torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
        torch.ones(one_hot.shape[0], 1, device=policy_device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()

    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    '''input_embeds = (one_hot @ embed_weights):
#这里是先假设one_hot形状为(batch_size, vocab_size)

    这是一个矩阵相乘的操作，其中
    one_hot
    是一个形状为(batch_size, vocab_size)
    的
    one - hot
    编码的张量，而
    embed_weights
    是一个形状为(vocab_size, embedding_dim)
    的嵌入权重矩阵。
    结果是一个形状为(batch_size, embedding_dim)
    的张量，其中包含了输入切片中每个
    token
    的嵌入表示。'''

    # now stitch it together with the rest of the embeddings
    # embeds = get_embeddings_t5(model, input_ids.unsqueeze(0)).detach()
    #
    # full_embeds = torch.cat(
    #     [
    #         embeds[:, :input_slice.start, :],  # 表示在整个输入序列中，选择从开头到 input_slice.start 之前的嵌入表示。
    #         input_embeds,
    #         embeds[:, input_slice.stop:, :]  # 目的是获取整个输入序列中，除了输入切片之外的部分的嵌入表示。
    #     ],
    #     dim=1)
    #
    # # .logits:从模型的输出中获取对数概率
    # logits = model(inputs_embeds=full_embeds).logits
    # targets = input_ids[target_slice]
    # loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)


         # actions:从vima中获得的字典,注意不是env.action_space
    output = []
    output.extend(actions["pose0_position"])
    output.extend(actions["pose0_rotation"])
    output.extend(actions["pose1_position"])
    output.extend(actions["pose1_rotation"])
    output = torch.Tensor(output)
    targets = torch.zeros([12], requires_grad=True)
    loss = nn.MSELoss()(output, targets)

    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)

    return grad




# def get_grad(actions):
#     """
#     Args:
#         actions:
#             (从vima中获得的字典,注意不是env.action_space)
#             {
#                 "pose0_position": [ , ], "pose0_rotation": [ , , , ],
#                 "pose1_position": [ , ], "pose1_rotation": [ , , , ]
#             }
#     """
#     # actions:从vima中获得的字典,注意不是env.action_space
#     output = []
#     output.extend(actions["pose0_position"])
#     output.extend(actions["pose0_rotation"])
#     output.extend(actions["pose1_position"])
#     output.extend(actions["pose1_rotation"])
#     output = torch.Tensor(output)
#     targets = torch.zeros([12], requires_grad=True)
#     loss = nn.MSELoss()(output, targets)
#     loss.backward()
#     print(loss.item())


# actions = {
#     "pose0_position": [0.4099999964237213, -0.15000000596046448],
#     "pose0_rotation": [0.0, 0.0, 0.0, 0.9600000381469727],
#     "pose1_position": [0.3700000047683716, 0.36000001430511475],
#     "pose1_rotation": [0.0, 0.0, 0.9600000381469727, -0.24000000953674316]
# }
# get_grad(actions)

def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    '''
    计算梯度的负值，然后在每行上选择前 topk 个最大的索引。就是从词表中取希望最大的256个
    '''

    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    '''
    取梯度下降最大的就相当于Peter来开密码锁，这一步相当于50个Peter同时来套密码
    '''

    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    '''
    
    '''

    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
    return cands


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    
    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), " 
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits
    

def forward(*, model, input_ids, attention_mask, batch_size=512):

    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        
        batch_input_ids = input_ids[i:i+batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask
    
    return torch.cat(logits, dim=0)

def target_loss(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])
    return loss.mean(dim=-1)


 #control_slice=suffix_manager._control_slice,
 #test_controls=new_adv_suffix,
def getlosses(tokenizer,input_ids,prefix_slice,test_controls, batch_size,policy_device):
#def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
#def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    # ... 省略了函数的参数声明

    if isinstance(test_controls[0], str):
        # 如果测试控制集合是字符串列表
        #max_len = control_slice.stop - control_slice.start
        max_len = prefix_slice.stop - prefix_slice.start
        # 对每个测试控制进行标记化，限制最大长度并转换为模型设备
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=policy_device)
            for control in test_controls
        ]
        # 寻找可用的填充标记，以便在下面创建嵌套张量
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        # 将测试控制转换为嵌套张量，并在需要时进行填充
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    #if not (test_ids[0].shape[0] == control_slice.stop - control_slice.start):
    if not (test_ids[0].shape[0] == prefix_slice.stop - prefix_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), "
            f"got {test_ids.shape}"
        ))

    # 生成位置索引，将测试控制插入到输入序列的指定切片中
    locs = torch.arange(prefix_slice.start, prefix_slice.stop).repeat(test_ids.shape[0], 1).to(policy_device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(policy_device),
        1,
        locs,
        test_ids
    )

    # 创建注意力掩码，将 pad_tok 之外的位置设为 1
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    # if return_ids:
    #     # 如果需要返回控制序列的 ID，则调用 forward 函数并返回 logits 以及控制序列的 ID
    #     del locs, test_ids;
    #     gc.collect()
    #     return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    # else:
    #     # 否则，只返回 logits
    #     del locs, test_ids
    #     logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
    #     del ids;
    #     gc.collect()
    #     return logits

    #
    losses =
    return losses

# 返回模型和分词器(tokenizer)
def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    # 下载模型参数
    # model_path:指定模型的本地路径或者Hugging Face模型Hub上的名称
    # trust_remote_code:否信任远程代码。
    # 如果设置为True，则允许从Hugging Face模型Hub上下载模型；如果设置为False，则不会下载远程模型，只会使用本地路径指定的模型。
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs
        ).to(device).eval()

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    # 根据受攻击的模型类型把分词器做相应变化
    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_policy_and_tokenizer(policy_path, tokenizer_path=None, device='cuda:0', **kwargs):
    policy = create_policy_from_ckpt(policy_path, device)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return policy, tokenizer