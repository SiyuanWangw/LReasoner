import nltk
import numpy as np
from tqdm import tqdm
import random
import json
from nltk.tree import Tree
# allennlp  transformer version==2.11.0
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")


condition_keywords = ["in order for", "thus"]
reverse_condition_keywords = ["due to", "owing to", "since", "if", "unless"]  #unless还要reverse negative
negative_words = ["not", "n't", "unable"]
reverse_negative_words = ["no", "few", "little", "neither", "none of"]
be_verbs = ["is", "are", "was", "were", "be"]
degree_indicator = ["only", "most", "best", "least"]

PRP_words = ["everyone", "someone", "anyone", "somebody", "everybody", "anybody"]


# judge whether have the keyword
def has_keyword(keyword_list, tokens):
    for each in keyword_list:
        if each in tokens:
            return tokens.index(each), each
        else:
            if len(nltk.word_tokenize(each)) > 1:
                token_str = " ".join(tokens)
                idx = token_str.find(each)
                if idx >= 0:
                    return len(nltk.word_tokenize(token_str[:idx])), each
    return -1, None


# judge whether two logical components are the same
def has_same_logical_component(vnp1, vnp2, vnp1_compo_tags):

    vnp1_tokens, vnp2_tokens = list(), list()
    for each_phrase in vnp1:
        vnp1_tokens.append([each[0] for each in each_phrase])
    for each_phrase in vnp2:
        vnp2_tokens.append([each[0] for each in each_phrase])
    # print(vnp1_tokens)
    # print(vnp2_tokens)

    vnp2_compo_tags = list()
    if len(vnp1_compo_tags) > 0:
        tag_record = max(vnp1_compo_tags)
    else:
        tag_record = -1
    for j in range(len(vnp2_tokens)):
        has_same = -1
        for i in range(len(vnp1_tokens)):
            vnp1_i = set(vnp1_tokens[i])
            vnp2_j = set(vnp2_tokens[j])

            # hyper-parameter: 0.7
            if len(vnp1_i & vnp2_j)/max(len(vnp2_j),1) > 0.5:
                has_same = vnp1_compo_tags[i]
                break
        if has_same == -1:
            has_same = tag_record + 1
            tag_record += 1

        vnp2_compo_tags.append(has_same)
    return vnp2_compo_tags


# identify the logical expression from noun and gerundial phrases
def identify_logical_expression(premise_vn_phrases):
    all_component_tags = list()

    if len(premise_vn_phrases) == 0:
        raise Exception("No premises")
    else:
        vnp1_compo_tags = list()
        for i in range(len(premise_vn_phrases[0])):
            vnp1_compo_tags.append(i)
        all_component_tags.append(vnp1_compo_tags)

        for j in range(1, len(premise_vn_phrases)):
            arg_1 = list()
            arg_3 = list()
            for k in range(j):
                arg_1 += premise_vn_phrases[k]
                arg_3 += all_component_tags[k]
            vnpj_compo_tags = has_same_logical_component(arg_1, premise_vn_phrases[j], arg_3)
            all_component_tags.append(vnpj_compo_tags)

        if len(premise_vn_phrases) > 7:
            print("Exception: more than 6 premises", len(premise_vn_phrases))

    return all_component_tags


def extract_logical_variables(conclusions, premises, answer_list):
    all_conc_vn_phrases, all_conc_negative_tags = list(), list()
    # print("Conclusion ......................................")
    for each_conclusion in conclusions:
        if each_conclusion is not None:
            conc_all_verb_nouns_phrases, conc_all_vnp_scales, conc_token_list = extract_np_vnp_constituents(each_conclusion)
            conc_vn_phrases, conc_negative_tags = identify_positive_negative_vnp(conc_all_verb_nouns_phrases, conc_all_vnp_scales, conc_token_list)
            # print(conc_vn_phrases)
            # print(conc_negative_tags)
            all_conc_vn_phrases.append(conc_vn_phrases)
            all_conc_negative_tags.append(conc_negative_tags)


    all_prem_vn_phrases, all_prem_negative_tags = list(), list()
    # print("Premise ......................................")
    for each_premise in premises:
        each_prem_all_verb_nouns_phrases, each_prem_all_vnp_scales, premise_token_list = extract_np_vnp_constituents(each_premise)
        each_prem_vn_phrases, each_prem_negative_tags = identify_positive_negative_vnp(each_prem_all_verb_nouns_phrases, each_prem_all_vnp_scales, premise_token_list)
        # print(each_prem_vn_phrases)
        # print(each_prem_negative_tags)
        all_prem_vn_phrases.append(each_prem_vn_phrases)
        all_prem_negative_tags.append(each_prem_negative_tags)

    all_ans_vn_phrases, all_ans_negative_tags = list(), list()
    # print("Answer ......................................")
    for each_answer in answer_list:
        each_ans_all_verb_nouns_phrases, each_ans_all_vnp_scales, answer_token_list =  extract_np_vnp_constituents(each_answer)
        each_ans_vn_phrases, each_ans_negative_tags = identify_positive_negative_vnp(each_ans_all_verb_nouns_phrases, each_ans_all_vnp_scales, answer_token_list)
        # print(each_ans_vn_phrases)
        # print(each_ans_negative_tags)
        all_ans_vn_phrases.append(each_ans_vn_phrases)
        all_ans_negative_tags.append(each_ans_negative_tags)

    return all_prem_vn_phrases, all_prem_negative_tags, \
           all_ans_vn_phrases, all_ans_negative_tags, \
           all_conc_vn_phrases, all_conc_negative_tags


# extract the noun and gerundial phrases from each constituent string
def extract_np_vnp_constituents(constituent_strs):
    all_np_vnp_phrases = list()
    all_np_vnp_scales = list()

    left_pare_num = constituent_strs.count('(')
    right_pare_num = constituent_strs.count(')')
    if left_pare_num > right_pare_num:
        constituent_strs = constituent_strs + ')'
    elif left_pare_num < right_pare_num:
        constituent_strs = '(' + constituent_strs
    constituent_trees = Tree.fromstring(constituent_strs)
    extracted_trees = recursive_extract_np_vnp(constituent_trees)

    i = 0
    while i < len(extracted_trees):
        each = extracted_trees[i]

        if each.label() == 'HYPH':
            if i > 0 and extracted_trees[i-1].label() == 'NP' and i < (len(extracted_trees)-1) and extracted_trees[i+1].label() == 'NP':
                poped_phrase = all_np_vnp_phrases.pop()
                all_np_vnp_phrases.append(poped_phrase + each.pos() + extracted_trees[i+1].pos())
                i += 1
        else:
            all_np_vnp_phrases.append(each.pos())
        i += 1

    cur_sent = " ".join(constituent_trees.leaves())
    for each_phrase in all_np_vnp_phrases:
        cur_phrase = " ".join([each[0] for each in each_phrase])
        # print(cur_phrase)
        idx = cur_sent.find(cur_phrase)
        start = len(nltk.word_tokenize(cur_sent[:idx]))
        all_np_vnp_scales.append([start, start+len(each_phrase)])
        # print(all_np_vnp_scales[-1])

    return all_np_vnp_phrases, all_np_vnp_scales, [each.lower() for each in constituent_trees.leaves()]


# recursively extract all noun and gerundial phrases from constituent trees
def recursive_extract_np_vnp(constituent_trees):
    extracted_trees = list()
    for each in constituent_trees:
        # print(each)
        if each.label() == 'NP':
            if each[0].label() == 'PRP' or (not isinstance(each[0][0], Tree) and each[0][0].lower() in PRP_words):
                if len(each) > 1:
                    extracted_trees.append(each[1])
                else:
                    pass
            elif each[0].label() == 'NP' and \
                    (each[0][0].label() == 'PRP' or (not isinstance(each[0][0][0], Tree) and each[0][0][0].lower() in PRP_words)):
                if len(each) > 1:
                    extracted_trees.append(each[1])
                else:
                    pass
            else:
                extracted_trees.append(each)
        elif each.label() == 'VP':
            if 'VB' in each[0].label():
                extracted_trees.append(each)
            else:
                for i in range(1, len(each)):
                    # print(each[i])
                    # if each[i].label() == 'VP':
                    extracted_trees += recursive_extract_np_vnp(each[i:i+1])
        elif each.label() == 'SBAR' or each.label() == 'S':
            # print('SBAR', len(each))
            extracted_trees += recursive_extract_np_vnp(each)
        elif each.label() == 'HYPH':
            extracted_trees.append(each)
        else:
            pass
            # print('Other', each)

    return extracted_trees


# identify whether a negative connective exists
def identify_positive_negative_vnp(all_vn_phrases, all_vn_scales, token_list):
    vn_phrases = list()
    negative_tags = list()
    index_start = 0

    for i in range(len(all_vn_phrases)):
        cur_neg_tag = False
        cur_reverse_tag = False
        cur_vnp_tokens = [each_token[0].lower() for each_token in all_vn_phrases[i]]
        cur_vn_phrase = all_vn_phrases[i]

        nega_kw_index, nega_kw = has_keyword(negative_words+reverse_negative_words, cur_vnp_tokens)
        if nega_kw_index >= 0:
            cur_neg_tag = True
            # print(cur_vnp_tokens)
            # print(nega_kw_index, nega_kw)
            if nega_kw in reverse_negative_words and nega_kw_index == 0:
                cur_reverse_tag = True
            cur_vn_phrase = cur_vn_phrase[: nega_kw_index] + cur_vn_phrase[nega_kw_index+len(nltk.word_tokenize(nega_kw)):]

        outer_nega_kw_index, outer_nega_kw = has_keyword(negative_words + reverse_negative_words, token_list[index_start: all_vn_scales[i][0]])
        if outer_nega_kw_index >= 0:
            if cur_neg_tag:
                cur_neg_tag = False
            else:
                cur_neg_tag = True

        index_start = all_vn_scales[i][1]
        vn_phrases.append(cur_vn_phrase)
        negative_tags.append([cur_neg_tag, cur_reverse_tag])

    reverse_negative_tags = list()
    j = 0
    while j < len(vn_phrases):
        if negative_tags[j][1]:
            reverse_negative_tags.append(bool(1-negative_tags[j][0]))
            if j+1 < len(vn_phrases):
                reverse_negative_tags.append(bool(1-negative_tags[j+1][0]))
            j += 1
        else:
            reverse_negative_tags.append(negative_tags[j][0])
        j += 1

    return vn_phrases, reverse_negative_tags


# identify the conditional relationship between two logical symbols
def identify_condition(all_vn_phrases, all_negative_tags):
    all_conditioned_vn_phrases, all_conditioned_negative_tags = list(), list()
    for i, each_sent_phrases in enumerate(all_vn_phrases):
        cur_sent_all_phrases = list()
        cur_sent_all_negative_tags = list()
        cur_sent_vn_reverse_tags = list()

        for j, cur_vn_phrase in enumerate(each_sent_phrases):
            token_list = [each[0].lower() for each in cur_vn_phrase]
            condi_kw_index, condi_kw = has_keyword(condition_keywords+reverse_condition_keywords, token_list)

            if condi_kw_index >= 0:
                if cur_vn_phrase[condi_kw_index-1][0] == ',':
                    cur_sent_all_phrases.append(cur_vn_phrase[:condi_kw_index-1])
                else:
                    cur_sent_all_phrases.append(cur_vn_phrase[:condi_kw_index])
                if condi_kw == 'unless':
                    cur_sent_all_negative_tags.append(bool(1-all_negative_tags[i][j]))
                else:
                    cur_sent_all_negative_tags.append(all_negative_tags[i][j])
                if condi_kw in reverse_condition_keywords:
                    cur_sent_vn_reverse_tags.append(True)
                else:
                    cur_sent_vn_reverse_tags.append(False)
                cur_sent_all_phrases.append(cur_vn_phrase[condi_kw_index + len(nltk.word_tokenize(condi_kw)):])
                cur_sent_all_negative_tags.append(all_negative_tags[i][j])
                cur_sent_vn_reverse_tags.append(False)
            else:
                cur_sent_all_phrases.append(cur_vn_phrase)
                cur_sent_all_negative_tags.append(all_negative_tags[i][j])
                cur_sent_vn_reverse_tags.append(False)

        reverse_all_vn_phrases = list()
        reverse_all_negative_tags = list()
        k = 0
        while k < len(cur_sent_all_phrases):
            if cur_sent_vn_reverse_tags[k]:
                if k + 1 < len(cur_sent_all_phrases):
                    reverse_all_vn_phrases.append(cur_sent_all_phrases[k+1])
                    reverse_all_vn_phrases.append(cur_sent_all_phrases[k])
                    reverse_all_negative_tags.append(cur_sent_all_negative_tags[k+1])
                    reverse_all_negative_tags.append(cur_sent_all_negative_tags[k])
                    k += 1
                else:
                    reverse_all_vn_phrases.append(cur_sent_all_phrases[k])
                    reverse_all_negative_tags.append(cur_sent_all_negative_tags[k])
            else:
                reverse_all_vn_phrases.append(cur_sent_all_phrases[k])
                reverse_all_negative_tags.append(cur_sent_all_negative_tags[k])
            k += 1
        # print(reverse_all_vn_phrases)
        # print(reverse_all_negative_tags)

        all_conditioned_vn_phrases.append(reverse_all_vn_phrases)
        all_conditioned_negative_tags.append(reverse_all_negative_tags)

    return all_conditioned_vn_phrases, all_conditioned_negative_tags


def spread_logical_expressions(all_vn_phrases, all_negative_tags, all_compo_tags):
    spread_all_vn_phrases = list()
    spread_all_negative_tags = list()
    spread_all_compo_tags = list()

    for i, each_vn in enumerate(all_vn_phrases):
        if len(each_vn) > 2:
            for j in range(1, len(each_vn)):
                spread_all_vn_phrases.append(each_vn[j-1: j+1])
                spread_all_negative_tags.append(all_negative_tags[i][j-1:j+1])
                spread_all_compo_tags.append(all_compo_tags[i][j-1:j+1])
        elif len(each_vn) == 2:
            spread_all_vn_phrases.append(each_vn)
            spread_all_negative_tags.append(all_negative_tags[i])
            spread_all_compo_tags.append(all_compo_tags[i])

    return spread_all_vn_phrases, spread_all_negative_tags, spread_all_compo_tags


# infer the entailed logical expressions
def infer_logical_expression(all_vn_phrases, all_negative_tags, all_compo_tags):
    all_logical_expressions = list()
    all_textual_vn_phrases = list()
    for i in range(len(all_compo_tags)):
        all_logical_expressions.append([[x, y] for x, y in zip(all_compo_tags[i], all_negative_tags[i])])
        all_textual_vn_phrases.append(all_vn_phrases[i])

    extended_logical_expression = list()
    extended_textual_vn_phrases = list()

    def reverse_logic(all_textual_vn_phrases, all_logical_expressions):
        cur_extended_logical_expression = list()
        cur_extended_textual_vn_phrases = list()

        for i in range(len(all_logical_expressions)):
            if len(all_logical_expressions[i]) == 2:
                rever_cur_logical_expression = [[x, bool(1-y)] for x, y in all_logical_expressions[i][::-1]]
                if rever_cur_logical_expression not in all_logical_expressions+cur_extended_logical_expression:
                    # all_logical_expressions.append(rever_cur_logical_expression)
                    # all_vn_phrases.append(all_vn_phrases[i][::-1])
                    cur_extended_logical_expression.append(rever_cur_logical_expression)
                    cur_extended_textual_vn_phrases.append(all_textual_vn_phrases[i][::-1])

        return cur_extended_logical_expression, cur_extended_textual_vn_phrases

    def transfer_logic(all_textual_vn_phrases, all_logical_expressions):
        # print("all + extended", "*"*100)
        # print(all_logical_expressions)
        # print(all_textual_vn_phrases)

        cur_extended_logical_expression = list()
        cur_extended_textual_vn_phrases = list()

        for i in range(len(all_logical_expressions)):
            other_logical_expressions = all_logical_expressions[:i]+all_logical_expressions[i+1:]
            other_textual_vn_phrases = all_textual_vn_phrases[:i]+all_textual_vn_phrases[i+1:]
            for j in range(len(other_logical_expressions)):
                if all_logical_expressions[i][1] == other_logical_expressions[j][0]:
                    trans_cur_logical_expression = [all_logical_expressions[i][0], other_logical_expressions[j][1]]
                    if trans_cur_logical_expression not in all_logical_expressions+cur_extended_logical_expression:
                        cur_extended_logical_expression.append(trans_cur_logical_expression)
                        cur_extended_textual_vn_phrases.append([all_textual_vn_phrases[i][0], other_textual_vn_phrases[j][1]])

        return cur_extended_logical_expression, cur_extended_textual_vn_phrases

    whether_continue = True

    while whether_continue:
        cur_rever_extended_logical_expression, cur_rever_extended_textual_vn_phrases = reverse_logic(all_textual_vn_phrases+extended_textual_vn_phrases, all_logical_expressions+extended_logical_expression)
        extended_logical_expression += cur_rever_extended_logical_expression
        extended_textual_vn_phrases += cur_rever_extended_textual_vn_phrases

        cur_trans_extended_logical_expression, cur_trans_extended_textual_vn_phrases = transfer_logic(all_textual_vn_phrases+extended_textual_vn_phrases, all_logical_expressions+extended_logical_expression)
        extended_logical_expression += cur_trans_extended_logical_expression
        extended_textual_vn_phrases += cur_trans_extended_textual_vn_phrases

        # print("Cur transfer extended", "*"*100)
        # print(extended_logical_expression)
        # print(extended_textual_vn_phrases)

        if len(cur_rever_extended_logical_expression)+len(cur_trans_extended_logical_expression)== 0:
            whether_continue = False

    return all_logical_expressions, all_textual_vn_phrases, extended_logical_expression, extended_textual_vn_phrases


# verbalize the logical expression into text
def logical_expression_to_text(logical_expression, vn_phrases):

    first_phrase, first_nega_tag = vn_phrases[0], logical_expression[0][1]
    second_phrase, second_nega_tag = vn_phrases[1], logical_expression[1][1]

    # first_verb_type = nltk.pos_tag([first_phrase[0][0]])[0][1]
    # second_verb_type = nltk.pos_tag([second_phrase[0][0]])[0][1]

    first_text = "If "
    if len(first_phrase) > 0:
        if 'VB' in first_phrase[0][1]:
            if first_phrase[0][1] == "VBZ":
                first_text += "it "
            else:
                first_text += "you "
        else:
            first_text += "it is "

        if first_nega_tag is True:
            if 'VB' in first_phrase[0][1]:
                if first_phrase[0][0] not in be_verbs:
                    if first_phrase[0][1] == "VBZ":
                        first_text += "does not " + ' '.join([each[0] for each in first_phrase]) + ","
                    elif first_phrase[0][1] == "VBD":
                        first_text += "did not " + ' '.join([each[0] for each in first_phrase]) + ","
                    else:
                        first_text += "do not " + ' '.join([each[0] for each in first_phrase]) + ","
                else:
                    first_text += first_phrase[0][0] + " not " + ' '.join([each[0] for each in first_phrase[1:]]) + ","
            else:
                first_text += "not " + ' '.join([each[0] for each in first_phrase]) + ","
        else:
            first_text += ' '.join([each[0] for each in first_phrase]) + ","

    second_text = "then "
    if len(second_phrase) > 0:

        if 'VB' in second_phrase[0][1]:
            if second_phrase[0][1] == "VBZ":
                second_text += "it will "
            else:
                second_text += "you will "
        else:
            second_text += "it will be "

        if second_nega_tag is True:
            if 'VB' in second_phrase[0][1]:
                if second_phrase[0][0] not in be_verbs:
                    second_text += "not " + ' '.join([each[0] for each in second_phrase]) + "."
                else:
                    second_text += "be not " + ' '.join([each[0] for each in second_phrase[1:]]) + "."
            else:
                second_text += "not " + ' '.join([each[0] for each in second_phrase]) + "."
        else:
            if second_phrase[0][0] not in be_verbs:
                second_text += ' '.join([each[0] for each in second_phrase]) + "."
            else:
                second_text += "be " + ' '.join([each[0] for each in second_phrase[1:]]) + "."

    return first_text + " " + second_text


# judge whether two logical components have overlap
def has_overlap_logical_component(answer_vnps, prem_vnps, answer_nega_tags, prem_nega_tags):
    # print(answer_vnps, answer_nega_tags)
    # print(prem_vnps, prem_nega_tags)
    for i in range(len(answer_vnps)):
        answ_vnp_i = set([each[0] for each in answer_vnps[i]])
        answ_degree_kw_index = has_keyword(degree_indicator, list(answ_vnp_i))[0]
        for j in range(len(prem_vnps)):
            prem_vnp_j = set([each[0] for each in prem_vnps[j]])

            # hyper-parameter: 0.7
            prem_degree_kw_index = has_keyword(degree_indicator,  list(prem_vnp_j))[0]
            if (answ_degree_kw_index >= 0 and prem_degree_kw_index < 0) \
                    or (answ_degree_kw_index < 0 and prem_degree_kw_index >= 0):
                return False
            if len(answ_vnp_i & prem_vnp_j)/max(len(answ_vnp_i),1) > 0.5:
                if answer_nega_tags[i] == prem_nega_tags[j]:
                    return True
    return False


# compute the overlap rate between two logical components
def has_overlap_logical_component_rate(answer_vnps, prem_vnps, answer_nega_tags, prem_nega_tags):
    overlap_num = 0
    for i in range(len(answer_vnps)):
        answ_vnp_i = set([each[0] for each in answer_vnps[i]])
        answ_degree_kw_index = has_keyword(degree_indicator, list(answ_vnp_i))[0]

        for j in range(len(prem_vnps)):
            prem_vnp_j = set([each[0] for each in prem_vnps[j]])

            prem_degree_kw_index = has_keyword(degree_indicator,  list(prem_vnp_j))[0]
            if (answ_degree_kw_index >= 0 and prem_degree_kw_index < 0) \
                    or (answ_degree_kw_index < 0 and prem_degree_kw_index >= 0):
                pass
            elif len(answ_vnp_i & prem_vnp_j)/max(len(answ_vnp_i),1) > 0.5:
                if answer_nega_tags[i] == prem_nega_tags[j]:
                    overlap_num += 1
                    break

    return overlap_num


# get verbalized extended context
def get_cur_all_extended_text(conclusion, premises, answer_list):
    all_extended_contexts = list()

    all_prem_vn_phrases, all_prem_negative_tags, \
    all_ans_vn_phrases, all_ans_negative_tags, \
    conc_vn_phrases, conc_negative_tags = extract_logical_variables(conclusion, premises, answer_list)

    all_prem_vn_phrases = all_prem_vn_phrases + conc_vn_phrases
    all_prem_negative_tags = all_prem_negative_tags + conc_negative_tags

    all_prem_vn_phrases, all_prem_negative_tags = identify_condition(all_prem_vn_phrases, all_prem_negative_tags)

    all_prem_compo_tags = identify_logical_expression(all_prem_vn_phrases)
    # print(all_prem_compo_tags)
    # print(len(all_prem_compo_tags))
    # if len(all_prem_vn_phrases) > 6:
    #     all_prem_vn_phrases = all_prem_vn_phrases[:6]
    #     all_prem_negative_tags = all_prem_negative_tags[:6]

    spread_all_vn_phrases, spread_all_negative_tags, spread_all_compo_tags = spread_logical_expressions(all_prem_vn_phrases, all_prem_negative_tags, all_prem_compo_tags)

    all_logical_expressions, all_textual_vn_phrases, extended_logical_expression, extended_textual_vn_phrases = infer_logical_expression(spread_all_vn_phrases, spread_all_negative_tags, spread_all_compo_tags)

    if len(extended_logical_expression) > 0:
        has_extended = 1
    else:
        has_extended = 0

    # answer_extend_num = list()
    # for j in range(len(all_ans_vn_phrases)):
    #     cur_answ_extended_contexts = list()
    #     for i in range(len(extended_logical_expression)):
    #         extended_negative_tags = [each[1] for each in extended_logical_expression[i]]
    #         whether_extend = has_overlap_logical_component_rate(all_ans_vn_phrases[j], extended_textual_vn_phrases[i],
    #                                                        all_ans_negative_tags[j], extended_negative_tags)
    #
    #         if whether_extend > 0:
    #             extended_text = logical_expression_to_text(extended_logical_expression[i],
    #                                                        extended_textual_vn_phrases[i])
    #             cur_answ_extended_contexts.append((extended_text, whether_extend))
    #
    #     sorted_cur_answ_extended_contexts = [each[0] for each in sorted(cur_answ_extended_contexts, key=lambda x: x[1], reverse=True)]
    #     answer_extend_num.append(len(sorted_cur_answ_extended_contexts))
    #
    #     if len(sorted_cur_answ_extended_contexts) == 0:
    #         all_extended_contexts.append("")
    #     elif len(sorted_cur_answ_extended_contexts) == 1:
    #         all_extended_contexts.append(sorted_cur_answ_extended_contexts[0])
    #     else:
    #         all_extended_contexts.append(
    #             sorted_cur_answ_extended_contexts[0] + " " + sorted_cur_answ_extended_contexts[1])
    #
    #     if len(sorted_cur_answ_extended_contexts) == 0:
    #         all_extended_contexts.append("")
    #     elif len(sorted_cur_answ_extended_contexts) == 1:
    #         all_extended_contexts.append(sorted_cur_answ_extended_contexts[0])
    #     elif len(sorted_cur_answ_extended_contexts) == 2:
    #         all_extended_contexts.append(sorted_cur_answ_extended_contexts[0] + " " + sorted_cur_answ_extended_contexts[1])
    #     else:
    #         all_extended_contexts.append(sorted_cur_answ_extended_contexts[0] + " " + sorted_cur_answ_extended_contexts[1] + " " + sorted_cur_answ_extended_contexts[2])

    answer_extend_num = list()
    for j in range(len(all_ans_vn_phrases)):
        cur_answ_extended_contexts = list()
        for i in range(len(extended_logical_expression)):
            extended_negative_tags = [each[1] for each in extended_logical_expression[i]]
            whether_extend = has_overlap_logical_component(all_ans_vn_phrases[j], extended_textual_vn_phrases[i], all_ans_negative_tags[j], extended_negative_tags)

            if whether_extend:
                extended_text = logical_expression_to_text(extended_logical_expression[i], extended_textual_vn_phrases[i])
                cur_answ_extended_contexts.append(extended_text)


        if len(cur_answ_extended_contexts) == 0:
            all_extended_contexts.append("")
        elif len(cur_answ_extended_contexts) == 1:
            all_extended_contexts.append(cur_answ_extended_contexts[0])
        else:
            sample_index = random.sample(range(len(cur_answ_extended_contexts)), 2)
            all_extended_contexts.append(cur_answ_extended_contexts[sample_index[0]] + " " + cur_answ_extended_contexts[sample_index[1]])

        # answer_extend_num.append(len(cur_answ_extended_contexts))
        # if len(cur_answ_extended_contexts) == 0:
        #     all_extended_contexts.append("")
        # elif len(cur_answ_extended_contexts) == 1:
        #     all_extended_contexts.append(cur_answ_extended_contexts[0])
        # elif len(cur_answ_extended_contexts) == 2:
        #     all_extended_contexts.append(cur_answ_extended_contexts[0] + " " + cur_answ_extended_contexts[1])
        # else:
        #     sample_index = random.sample(range(len(cur_answ_extended_contexts)), 3)
        #     all_extended_contexts.append(cur_answ_extended_contexts[sample_index[0]] + " " + cur_answ_extended_contexts[sample_index[1]] + " " + cur_answ_extended_contexts[sample_index[2]])

    max_length = max([len(each) for each in all_extended_contexts])

    return all_extended_contexts, (max_length, has_extended, sum(answer_extend_num)), \
           (all_logical_expressions, all_textual_vn_phrases, extended_logical_expression, extended_textual_vn_phrases)


desired_question_type_list = [
    "Identify the conclusion", "Identify the role", "Point at issue and disagreement",
    "Must be true or Cannot be true", "Most strongly supported", "Complete the passage",
    "Necessary assumption", "Sufficient assumption", "Strengthen", "Weaken", "Useful to know to evaluate", "Explain or Resolve",
    # "Identify the technique", "Parallel reasoning", "Identify the flaw", "Parallel flaw",
    # "Identify the principle", "Parallel principle",
]

entailment_question_type_list = [
    "Identify the conclusion", "Identify the role", "Point at issue and disagreement",
    "Must be true or Cannot be true", "Most strongly supported", "Complete the passage",
]

assumption_question_type_list = [
    "Necessary assumption", "Sufficient assumption", "Strengthen", "Weaken", "Useful to know to evaluate", "Explain or Resolve",
]

def save_all_extended_context(data_type=0):
    if data_type == 0:
        all_context_constituents_file = '../reclor-data/train_context_constituents_file.npy'
        all_options_constituents_file = '../reclor-data/train_options_constituents_file.npy'
        # extended_logic_file = '../reclor-data/train_extended_logic.npy'
        extended_context_file = '../reclor-data/train_extended_context_cp_v5.npy'
    elif data_type == 1:
        all_context_constituents_file = '../reclor-data/val_context_constituents_file.npy'
        all_options_constituents_file = '../reclor-data/val_options_constituents_file.npy'
        # extended_logic_file = '../reclor-data/val_extended_logic.npy'
        extended_context_file = '../reclor-data/val_extended_context_cp_v5.npy'
    else:
        all_context_constituents_file = '../reclor-data/test_context_constituents_file.npy'
        all_options_constituents_file = '../reclor-data/test_options_constituents_file.npy'
        # extended_logic_file = '../reclor-data/test_extended_logic.npy'
        extended_context_file = '../reclor-data/test_extended_context_cp_v5.npy'

    # v3: same for all question type (prem) + extend num 2
    # v4: same for all question type (prem) + extend num 3

    # v5: same for all question type (prem + conclusion) + extend num 2
    # v6: same for all question type (prem + conclusion) + extend num 3

    # v7: same for all question type (prem + conclusion) + extend num 2, rank related extension
    # v8: same for all question type (prem + conclusion) + extend num 3, rank related extension

    all_extended_contexts = list()
    # all_extended_logic = list()

    conclusions_premises_constituents = np.load(all_context_constituents_file)
    answers_constituents = np.load(all_options_constituents_file)

    has_extend_instance_num = 0
    has_extend_context_num = 0
    answer_extend_num = list()

    for i in tqdm(range(len(conclusions_premises_constituents))):
        conclusions = []
        premises = conclusions_premises_constituents[i]
        answer_list = answers_constituents[i]

        cur_all_extended_context, max_length, logic_result  = get_cur_all_extended_text(conclusions, premises, answer_list)
        # all_extended_logic.append(logic_result)

        if max_length[0] > 0:
            has_extend_instance_num += 1
        if max_length[1] > 0:
            has_extend_context_num += 1
        answer_extend_num.append(max_length[2])

        all_extended_contexts.append(cur_all_extended_context)

    print(has_extend_instance_num, has_extend_context_num, np.mean(answer_extend_num)/4)

    np.save(extended_context_file, all_extended_contexts)
    # np.save(extended_logic_file, all_extended_logic)


# get all constituents in the text using allennlp
def get_all_constituents(data_type=0):
    if data_type == 0:
        lr_file = '../reclor-data/train.json'
        all_context_constituents_file = '../reclor-data/train_context_constituents_file.npy'
        all_options_constituents_file = '../reclor-data/train_options_constituents_file.npy'
    elif data_type == 1:
        lr_file = '../reclor-data/val.json'
        all_context_constituents_file = '../reclor-data/val_context_constituents_file.npy'
        all_options_constituents_file = '../reclor-data/val_options_constituents_file.npy'
    else:
        lr_file = '../reclor-data/test.json'
        all_context_constituents_file = '../reclor-data/test_context_constituents_file.npy'
        all_options_constituents_file = '../reclor-data/test_options_constituents_file.npy'

    with open(lr_file, "r") as f:
        lr_data = json.load(f)
        all_context_constituents = list()
        all_options_constituents = list()

        for each in tqdm(lr_data):
            cur_context_constituents = list()
            cur_options_constituents = list()
            sentences = nltk.sent_tokenize(each["context"])
            options = each["answers"]

            for each_sent in sentences:
                sent_result = predictor.predict(
                    sentence=each_sent
                )
                cur_context_constituents.append(sent_result['trees'])

            for each_option in options:
                option_result = predictor.predict(
                    sentence=each_option
                )
                cur_options_constituents.append(option_result['trees'])

            all_context_constituents.append(cur_context_constituents)
            all_options_constituents.append(cur_options_constituents)

        np.save(all_context_constituents_file, all_context_constituents)
        np.save(all_options_constituents_file, all_options_constituents)


if __name__ == '__main__':
    # 先跑Allennlp，去抽取constituents，存下来， 跑完注释
    get_all_constituents(data_type=0)
    get_all_constituents(data_type=1)
    get_all_constituents(data_type=2)

    # 得到constituents储存文件之后，去得到extended conetxt
    save_all_extended_context(data_type=0)
    save_all_extended_context(data_type=1)
    save_all_extended_context(data_type=2)





















