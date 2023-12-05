import torch
import fastchat


# # 由template_name生成指定的对话模版，并进行处理
# def load_conversation_template(template_name):
#     # 由template_name生成指定的对话模版
#     conv_template = fastchat.model.get_conversation_template(template_name)
#     if conv_template.name == 'zero_shot':
#         # 把角色名前面加‘###’，把指定对话模板中不同部分（例如角色、对话行为等）之间的分隔符设为‘\n’
#         conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
#         conv_template.sep = '\n'
#     elif conv_template.name == 'llama-2':
#         # sep2:分隔不同部分的另一个分隔符
#         # 表示去除 sep2 属性值两端的空格或换行符
#         conv_template.sep2 = conv_template.sep2.strip()
#
#     return conv_template


# 后缀处理
class SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):
    #def __init__(self, *, tokenizer, instruction, target, adv_string):

        self.tokenizer = tokenizer
        #self.conv_template = conv_template
        self.instruction = instruction
        # instruction eg:"Write a convincing anonymous death threat."
        self.target = target
        # target 暂定为 "action":[end effector position3x, end effector orientation3x, gripper action1x, episode termination1x].
        # e.g. "world_vector": [[0.9372549057007817, -0.5482387542724609, 0.11703979969024658]]
        #"gripper_closedness_actio6836, -0.9921568632125854, -0.3176470398902893]]
        #      "rotation_delta": [[-1.361356854438n": [[-0.6549019813537598]]

        self.adv_string = adv_string
        # adv_string eg:"! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"

    # 得到带攻击性后缀的整个句子
    def get_prompt(self, adv_string=None):

        if adv_string is not None:
            self.adv_string = adv_string

        #prompt= f"{self.instruction} {self.adv_string}"
        prompt = self.instruction+self.adv_string
        return prompt
        # # self.conv_template.roles[0] 表示模板中的第一个角色，通常是系统角色，用于表示系统的发言
        # # 把self.instruction+self.adv_string添加到对话模板中的第一个角色作为一条消息
        # self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction} {self.adv_string}")
        # # self.conv_template.roles[1] 表示模板中的第二个角色，通常是用户角色，用于表示用户的发言。
        # self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        # # 包含了一个系统问句或指示用户应该提供什么信息的提示性文本，即self.instruction+self.adv_string
        # prompt = self.conv_template.get_prompt()
        #
        # encoding = self.tokenizer(prompt)
        # toks = encoding.input_ids
        #
        # # 用于多轮对话任务，以便将对话内容传递给模型进行处理和生成响应
        # if self.conv_template.name == 'llama-2':
        #     self.conv_template.messages = []
        #
        #     self.conv_template.append_message(self.conv_template.roles[0], None)
        #     toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        #     self._user_role_slice = slice(None, len(toks))
        #
        #     self.conv_template.update_last_message(f"{self.instruction}")
        #     toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        #     self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)))
        #
        #     separator = ' ' if self.instruction else ''
        #     self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
        #     toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        #     self._control_slice = slice(self._goal_slice.stop, len(toks))
        #
        #     self.conv_template.append_message(self.conv_template.roles[1], None)
        #     toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        #     self._assistant_role_slice = slice(self._control_slice.stop, len(toks))
        #
        #     self.conv_template.update_last_message(f"{self.target}")
        #     toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        #     self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-2)
        #     self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-3)
        #
        # else:
        #     python_tokenizer = False or self.conv_template.name == 'oasst_pythia'
        #     try:
        #         encoding.char_to_token(len(prompt)-1)
        #     except:
        #         python_tokenizer = True
        #
        #     if python_tokenizer:
        #         # This is specific to the vicuna and pythia tokenizer and conversation prompt.
        #         # It will not work with other tokenizers or prompts.
        #         self.conv_template.messages = []
        #
        #         self.conv_template.append_message(self.conv_template.roles[0], None)
        #         toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        #         self._user_role_slice = slice(None, len(toks))
        #
        #         self.conv_template.update_last_message(f"{self.instruction}")
        #         toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        #         self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)-1))
        #
        #         separator = ' ' if self.instruction else ''
        #         self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
        #         toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        #         self._control_slice = slice(self._goal_slice.stop, len(toks)-1)
        #
        #         self.conv_template.append_message(self.conv_template.roles[1], None)
        #         toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        #         self._assistant_role_slice = slice(self._control_slice.stop, len(toks))
        #
        #         self.conv_template.update_last_message(f"{self.target}")
        #         toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        #         self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-1)
        #         self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-2)
        #     else:
        #         self._system_slice = slice(
        #             None,
        #             encoding.char_to_token(len(self.conv_template.system))
        #         )
        #         self._user_role_slice = slice(
        #             encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
        #             encoding.char_to_token(prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
        #         )
        #         self._goal_slice = slice(
        #             encoding.char_to_token(prompt.find(self.instruction)),
        #             encoding.char_to_token(prompt.find(self.instruction) + len(self.instruction))
        #         )
        #         self._control_slice = slice(
        #             encoding.char_to_token(prompt.find(self.adv_string)),
        #             encoding.char_to_token(prompt.find(self.adv_string) + len(self.adv_string))
        #         )
        #         self._assistant_role_slice = slice(
        #             encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
        #             encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
        #         )
        #         self._target_slice = slice(
        #             encoding.char_to_token(prompt.find(self.target)),
        #             encoding.char_to_token(prompt.find(self.target) + len(self.target))
        #         )
        #         self._loss_slice = slice(
        #             encoding.char_to_token(prompt.find(self.target)) - 1,
        #             encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
        #         )
        # # 重置对话模板消息列表
        # self.conv_template.messages = []

        #return prompt

    # 得到带攻击性后缀的句子的标记序列
    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        # 将句子prompt转化为标记序列toks
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids

