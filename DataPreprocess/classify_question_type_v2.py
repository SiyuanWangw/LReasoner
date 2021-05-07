import json
import numpy as np

question_type_list = [
    "Identify the conclusion", "Identify the role", "Dispute",
    "Entailment", "Most strongly supported", "Complete the passage",
    "Necessary assumption", "Sufficient assumption", "Strengthen", "Weaken", "Evaluate", "Paradox",
    "Identify the technique", "Parallel reasoning", "Identify the flaw", "Parallel flaw",
    "Identify the principle", "Parallel principle",
]

question_type_tag_dict ={
    "Identify the conclusion": ['main point', 'main conclusion', 'conclusion', 'inference (main point)'],
    "Identify the role": ['method (statement)', 'method (role of a statement)', 'role of a statement', 'role of statement'],
    "Dispute": ['point at issue', 'inference [point at issue]', 'point', 'inference (point of disagreement)', 'point-at-issue', 'inference (point at issue)'],
    "Entailment": ['must be true', 'inference (formal logic)', 'cannot be true', 'inference (formal logic)/all except', 'inference except (formal logic)', 'inference / could be true except', 'inference (could be true except)', 'inference except'],
    "Most strongly supported": ['most strongly supported'],
    "Complete the passage": ['complete the passage', 'inference [fill in the blank]', 'inference (fill-in)'], # 分为两种：conclusion和premise
    "Necessary assumption": ['necessary assumption', 'assumption (necessary)', ],
    "Sufficient assumption": ['sufficient assumption', 'assumption (sufficient)'],
    "Strengthen": ['strengthen', 'strengthen the argument', 'strengthen the argument (all except)', 'strengthen except', 'strengthen (except)', 'strengthen / all except', 'strengthen the argument/all except', 'strengthen the argument / except', 'strengthen / except', 'strengthen/all except'],
    "Weaken": ['weaken', 'weakening', 'weaken the argument', 'principle (identify/weaken)', 'weaken (except)', 'weaken the argument except', 'weaken the argument (all except)', 'weaken the argument / all except'],
    "Evaluate": ['evaluate', 'strengthen/weaken (evaluate the argument)', 'strengthen/weaken (evaluate the argument/least)'],
    "Paradox": ['paradox', 'resolve/explain', 'paradox / all except', 'paradox except', 'paradox (all except)', 'paradox/all except', 'paradox / except'],
    "Identify the technique": ['method (argument)', 'method of argument', 'method of argumen'],
    "Parallel reasoning": ['parallel (reasoning)', 'parallel reasoning', 'parallel', 'parallel reasoning (formal logic)'],
    "Identify the flaw": ['flaw', 'logical flaw', 'logical flawflaw', 'flaw (formal logic)', 'logical flaw (formal logic)', 'faulty logic'],
    "Parallel flaw": ['parallel flaw', 'parallel (flaw)', 'parallel reasoning (flaw)', 'parallel reasoning (logical flaw)', 'parallel reasoning (flaw, formal logic)', 'parallel reasoning (flaw; formal logic)'],
    "Identify the principle": ['principle (identify)', 'principle (identify/strengthen)', 'principle (all except)'],
    "Parallel principle": ['principle (apply)', 'principle (apply/inference)'],
}

question_type_keyword_dict = {
    "Parallel flaw": [
        [
            ["most similar", "flaw"], ["most similar", "questionable"], ["most closely resemble", "flaw"],
            ["parallel", "flaw"], ["similar", "error"], ["same kind", "error"], ["most like", "flaw"], ["most similar", "erroneous"],
            ["parallel", "faulty"], ["most similar", "faulty"], ["exhibits both of the logical flaws"], ['analogy', 'flaw']
        ],
        []
    ],
    "Identify the flaw": [
        [
            ["is flawed because"], ["is questionable because"], ["is ineffective because"], ["is not sound because"], ["is in error because"], ["is inadequate because"],
            ["most vulnerable to"], ["vulnerable to" "criticism"], ["identifies", "error"], ["identifies", "problem"], ["reasoning is flawed"],
            ["indicate", "flaw"], ["indicate", "flaw"], ["describe", "flaw"], ["describe", "error"], ["express", "criticism"],
            ["falls short of"], ["failure to do"], ["misinterpret"], ["misleading"], ["misunderstanding"],
            ["which one of the following", "weakness"], ["which one of the following is", "criticism"], ["which one of the following", "error"],
            ["which one of the following criticisms"], ["which one of the following is", "flaw"],
            ["which of the following", "weakness"], ["which of the following is", "criticism"], ["which of the following", "error"],
            ["which of the following criticisms"], ["which of the following is", "flaw"],
            ["flaw", "is that"], ["weakness", "is that"], ["error", "is that"],  ["questionable", "technique"], ["never provide conclusive proof"]
        ],
        []
    ],
    "Parallel reasoning": [
        [
            ["most similar", "pattern of reasoning"], ["most closely similar", "pattern of reasoning"], ["most closely parallel", "pattern of reasoning"],
            ["most parallel", "pattern of reasoning"], ["most closely parallel", "reasoning"], ["most similarly to"],
            ["most similar", "logical structure"], ["most like", "logical structure"], ["most closely parallel", "argumentative structure"],
            ["most similar", "logical feature"], ["most closely parallel", "logical feature"], ["most resemble", "logical feature"],
            ["most nearly similar", "situation"], ["most closely parallel", "the way"], ["most closely parallel", "in its reasoning"],
            ["most similar in its reasoning"], ["most parallel in its reasoning"], ["most closely resemble", "reasoning"],
            ["preserve the force of"], ["analysis", "most appropriate"], ["most similar to", "the reasoning"]
        ],
        []
    ],
    "Identify the technique": [
        [
            ["argument", "proceeds by"], ["technique of reasoning"], ["express", "the method"], ["argumentative strategies"], ["argumentative strategy"],
            ["argumentative technique"], ["reasoning techniques"], ["techniques of reasoning"], ["in responding to"], ["persuade by"],
            ["strategies in criticizing"], ["the executive’s reasoning does which one of the following"], ["the executive’s reasoning does which of the following"],
            ["method of", "argument is to"], ["responds to", "by"], ["reply to", "by"], ["response to", "characterize"], ["response to", "argument is that"],
            ["counter", "argument by"], ["criticize", "by"], ["bases", "on which one of the following"], ["bases", "on which of the following"], ["counter", "use", "technique"],
            ["how", "response", "related to"], ["in attempting to refute"], ["response to", "does which one of the following"], ["response to", "does which of the following"],
            ["response to", "uses which one of the following"], ["response to", "uses which of the following"], ["object", "argument by"], ["method of persuasion"], ["method of reasoning"],
            ["rejoinder", "proceeds by"], ["describes the relationship between", "argument"], ["against", "proceeds by"], ["how", "proceed"],
            ["related to", "in which one of the following ways"], ["related to", "in which of the following ways"], ["responds to", "in which one of the following ways"],
            ["responds to", "in which of the following ways"], ["the relationship of", "is that"]
        ],
        ["in her argument,", "in the passage, the author", "in order to advance her point of view", " by", #"in the argument,",
         "the method of the argument is to", "does which one of the following?", "in stating the argument,", "the argument seeks to do which",
         "in advancing his argument?", "counters the objection by", "challenges that claim by", "strategy on the grounds that"]
    ],
    "Necessary assumption": [
        [
            ["assumption", "depend"], ["assuming", "depend"], ['assuming ', 'relies'], ['assumption', 'relies'], ['assumption', 'rely'],
            ["assumption", " base"], ['assume', 'require'], ["assumption", "require"], ["assumes which one of the following"], ["assumes which of the following"],
            ["presupposes which one of the following"], ["presupposes which of the following"], ["makes which one of the following assumption"],
            ["makes which of the following assumption"], ["makes the assumption that"], ["is supported only"],
            ['assumption', 'necessary'], ["would be necessary to"], ["make", "of the following assumption"], ["must", "assume in order to conclude that"],
        ],
        ["requires the assumption that", "which one of the following is assumed", "which of the following is assumed",
         "which one of the following is an assumption of the argument", "which one of the following is an assumption made",
         "which of the following is an assumption of the argument", "which of the following is an assumption made", "assumes that",
         "have to be auusmed that", "which one of the following must be assumed", "which of the following must be assumed", "presupposes that"
         ]
    ],
    "Sufficient assumption": [
        [
            ["if which one of the following is assumed"], ["given if which one of the following is true"],
            ["if which of the following is assumed"], ["given if which of the following is true"], ["argument to be properly inferred"],
            ["assume", "conclusion", "properly drawn"], ["assumption", "conclusion", "properly drawn"], ["assumption", "sufficient"],
            ["does the conclusion logically follow"], ["assumption", "make the conclusion", "logical"], ["assume", "follows logically"],
            ["if assumed", "draw", "conclusion"], ["enable", "to be properly"], ["assumption", "conclusion", "justify"],
            # ["if true", "draw", "conclusion"], ["true", "conclusion", "properly drawn"],
        ],
        ["if one knows that", "if one also knows that"]
    ],
    "Strengthen": [
        [
            ["which one of the following", "if true", "strengthen"], ["which of the following", "if true", "strengthen"],
            ["which one of the following", "is a statement", "address the point"], ["which of the following", "is a statement", "address the point"],
            ["which one of the following, if true", "support"], ["which of the following, if true", "support"],
            ["which one of the following reasons, if true", "support"], ["which of the following reasons, if true", "support"],
            ["which one of the following proposals, if true", "support"], ["which of the following proposals, if true", "support"],
            ["which one of the following statements, if true", "support"], ["which of the following statements, if true", "support"],
            ["which one of the following", "would strengthen", "conclusion"], ["which of the following", "would strengthen", "conclusion"],
            ["which one of the following, if also true", "support"], ["which of the following, if also true", "support"],
            ["which one of the following", "if also true", "help to justify"], ["which of the following", "if also true", "help to justify"],
            ["which one of the following", "if true", "help to justify"], ["which of the following", "if true", "help to justify"],
            ["which one of the following", "add", "helps to justify"], ["which of the following", "add", "helps to justify"],
            ["which one of the following", "if it were determined to be true", "evidence"], ["which of the following", "if it were determined to be true", "evidence"],
            ["which one of the following", "if it occurred", "evidence"], ["which of the following", "if it occurred", "evidence"],
            ["if which one of the following were true", "strengthen"], ["if which of the following were true", "strengthen"],
            ["would be least supported", "by the truth of which one of the following"], ["would be least supported", "by the truth of which of the following"],
            ["which one of the following", "if implemented", "improve"], ["which of the following", "if implemented", "improve"],
            ["each of the following, if true", "support"], ["would strongly favor"], ["each of the following", "if true", "strengthen"],
            ["most to justify the conclusion"], ["helps to justify the reasoning"], ["most contribute to a justification"],
            ["provide", "strongest evidence"], ["provide", "strongest grounds"]
        ],
        []
    ],
    "Weaken": [
        [
            ["which one of the following", "if true", "undermine"], ["which one of the following", "if true", "cast", "doubt"],
            ["which one of the following", "strongest counter to"], ["which one of the following", "if true", "weaken"],
            ["which one of the following", "if true", "challenge the conclusion"], ["which one of the following", "if true", "call", "into question"],
            ["which one of the following", "if true", "defense against"], ["which one of the following", "if true", "invalidate"],
            ["which one of the following", "if true", "counter"], ["which one of the following", "if true", "damaging"],
            ["which one of the following", "if true", "contribute", "to a refutation"], ["which one of the following", "if true", "indicates a weakness"],
            ["if which one of the following is true", "misleading"], ["which one of the following", "if true", "argue", "against"],
            ["which one of the following", "if true", "without support"], ["which one of the following", "if accepted", "require him to reconsider"],
            ["which one of the following", "if true", "incomplete"],  ["which one of the following", "provide", "reason", "exercising caution"],
            ["which one of the following", "if true", "limits the effectiveness"],
            ["which one of the following", "if true", "indicate", "would not"], ["which one of the following", "if true", "challenging the conclusion"],
            ["which of the following", "if true", "undermine"], ["which of the following", "if true", "cast", "doubt"],
            ["which of the following", "strongest counter to"], ["which of the following", "if true", "weaken"],
            ["which of the following", "if true", "challenge the conclusion"], ["which of the following", "if true", "call", "into question"],
            ["which of the following", "if true", "defense against"], ["which of the following", "if true", "invalidate"],
            ["which of the following", "if true", "counter"], ["which of the following", "if true", "damaging"],
            ["which of the following", "if true", "contribute", "to a refutation"], ["which of the following", "if true", "indicates a weakness"],
            ["if which of the following is true", "misleading"], ["which of the following", "if true", "argue", "against"],
            ["which of the following", "if true", "without support"], ["which of the following", "if accepted", "require him to reconsider"],
            ["which of the following", "if true", "incomplete"],  ["which of the following", "provide", "reason", "exercising caution"],
            ["which of the following", "if true", "limits the effectiveness"], ["which of the following piece", "evidence", "cast", "doubt"],
            ["which of the following", "if true", "indicate", "would not"], ["which of the following", "if true", "challenging the conclusion"],
            ["counters", "objection"], ["call into question", "if"], ["calls into question", "if"], ["called into question", "if"], ["calls in question", "if"],
            ["each of the following", "if true", "challenge"], ["undermine", "if it were true that"], ["weaken", "if it is true that"],
            ["each of the following", "if true", "cast", "doubt on"], ["each of the following", "if true", "weaken"],
        ],
        ["would not follow if the"]
    ],
    "Evaluate": [
        [
            ["most helpful", " evaluat"], ["least helpful", " evaluat"], ["most helpful", "judge whether"], ["most helpful", "determining whether"],
            ["most useful", " evaluat"], ["least useful", " evaluat"], ["most important", " evaluat"], ["least important", " evaluat"],
            ["most important", "decide"], ["most relevant", " evaluat"],
        ],
        []
    ],
    "Paradox": [
        [
            ["which one of the following", "if true", "explain"], ["which one of the following", "if true", "resolve"],
            ["if which one of the following were true", "explanation", "persuasive"],
            ["which one of the following", "if true", "resolving"], ["which one of the following", "if true", "resolution"],
            ["which one of the following", "if true", "account for"], ["which one of the following", "if true", "reconcile"],
            ["which one of the following", "if true", "establish", "reason"], ["which one of the following", "helps to account for"],
            ["which one of the following", "if true", "justify", "paradoxical"], ["which one of the following", "if true", "explanation"],
            ["which one of the following", "most helps account for"], ["which one of the following", "if true", "show", "correct"],
            ["which one of the following", "most helps to explain"],
            ["which of the following", "if true", "explain"], ["which of the following", "if true", "resolve"],
            ["if which of the following were true", "explanation", "persuasive"],
            ["which of the following", "if true", "resolving"], ["which of the following", "if true", "resolution"],
            ["which of the following", "if true", "account for"], ["which of the following", "if true", "reconcile"],
            ["which of the following", "if true", "establish", "reason"], ["which of the following", "helps to account for"],
            ["which of the following", "if true", "justify", "paradoxical"], ["which of the following", "if true", "explanation"],
            ["which of the following", "most helps account for"], ["which of the following", "if true", "show", "correct"],
            ["which of the following", "most helps to explain"],
            ["each of the following", "if true", "explain"], ["each of the following", "if true", "explanation"],
            ["each of the following", "helps to account for"], ["provides", "reason"], ["each of the following", "resolve"],
        ],
        []
    ],
    "Entailment": [
        [
            ["which one of the following", "logically follows from"], ["which one of the following", "follows logically from"],
            ["which one of the following", "inferred from"], ["which one of the following", "drawn from"], ["which one of the following", "proper inference from"],
            ["which one of the following", "concluded from"], ["which one of the following", "must", "be true", "on the basis of"],
            ["if", "are true", "which one of the following", "must", "be true"], ["if", "are true", "which one of the following", "can", "be drawn"],
            ["if", "are true", "which one of the following", "can", "inferred from"], ["if", "are true", "which one of the following", "can be properly inferred"],
            ["if", "is correct", "which one of the following", "can", "be drawn"], ["the statements above", "logically commit", "which one of the following"],
            ["assume", "be true", "which one of the following", "must", "be true"], ["which one of the following", "conflicts with"],
            ["which one of the following", "must be false"], ["if", "is accurate", "cannot be", "which one of the following"],
            ["which one of the", "does not meet", "requirement"], ["which one of the following", "cannot be true"],
            ["which one of the following", "drawn on the basis of"], ["commit", "which one of the following positions"],
            ["which of the following", "logically follows from"], ["which of the following", "follows logically from"],
            ["which of the following", "inferred from"], ["which of the following", "drawn from"], ["which of the following", "proper inference from"],
            ["which of the following", "concluded from"], ["which of the following", "must", "be true", "on the basis of"],
            ["if", "are true", "which of the following", "must", "be true"], ["if", "are true", "which of the following", "can", "be drawn"],
            ["if", "are true", "which of the following", "can", "inferred from"], ["if", "are true", "which of the following", "can be properly inferred"],
            ["if", "is correct", "which of the following", "can", "be drawn"], ["the statements above", "logically commit", "which of the following"],
            ["assume", "be true", "which of the following", "must", "be true"], ["which of the following", "conflicts with"],
            ["which of the following", "must be false"], ["if", "is accurate", "cannot be", "which of the following"],
            ["which of the", "does not meet", "requirement"], ["which of the following", "cannot be true"],
            ["which of the following", "drawn on the basis of"], ["commit", "which of the following positions"],
            ["if", "are true", "each of the following", "could", "be true"], ["if", "are true", "each of the following", "must", "be true"],
            ["each of the following", "inferred from"], ["of the following, which one", "follows logically from"], ["provide", "incentive", "to do each of the following"],
            ["interpret", "to imply that"], ["inferred from", "if the press is"], ["hold inconsistent beliefs", "if", "believe that"],
            ["if", "true", "each of the following", "could", "be true"], ["according to the argument", "each of the following", "could", "be true"],
            ["if accurate", "evidence against"], ["assuming", "are accurate", "cannot be true"],  ["if", "accurate", "each of the following", "could be true"],
            # ["be", "concluded", "that"],
        ],
        ["which one of the following can be expected as a result", "lead to which one of the following conclusions?", "committed to which one of the following?",
         "which of the following can be expected as a result", "lead to which of the following conclusions?", "committed to which of the following?",
         "which one of the following can be expected as a result?", "which of the following can be expected as a result?",
         "it can be properly inferred that", "a consequence of the view above is that", "to lead to the conclusion that",
         "leads to the conclusion that", "what cannot be true?"],
    ],
    "Most strongly supported": [
        [
            ["most strongly support", "the information above"], ["most reasonably be concluded", "the information above"], ["the information above", "least support"],
            ["most logically completes the passage"], ["the information above", "most support"], ["the statements above", "support the view"],
            ["the statements above", "support which"], ["the statements above", "support", "except"],
            ["provide strongest support", "the statements above"], ["most strongly support", "the statements above"], ["most support", "the statements above"],
            ["provides a reason for accepting", "the view above"], ["the analogy above", "best understood"], ["the observations above", "provide most evidence"],
            ["the parasitic-connection hypothesis", "most strongly support"], ["by the information in the passage", "support"], ["most strongly support", "by the results of"],
            ["the circumstances described above", "most strongly support"], ["if", "also holds that"], ["by the information above", "support"],
            ["the proposal above", "most likely to result in"], ["best support", "the statements above"], ["the statements in the passage", "most strongly support"],
            ["provide the strongest support for which one of the following"], ["provide the strongest support for which of the following"],
            ["is supported by the passage"], ["serve", "least", "evidence"], ["provides grounds for", "accept"], ["used as part of an argument"]
        ],
        ["the passage provides the most support", "the passage as a whole provides the most support"]
    ],
    "Complete the passage": [
        [
            ["completes the", "passage"], ["completes the", "argument"], ["completes the", "explanation"], ["logically concludes the argument"],
            ["completes the", "conclusion"], ["completion for the argument"], ["completes the", "paragraph"], ["completion of the paragraph"],
        ],
        []
    ],
    "Dispute": [
        [
            ["disagree", "over"], ["disagree", "about"], ["point at issue between"], ["disagreement between"],
            ["response to", "is structured to demonstrate"], ["disagree on whether"], ["issue", "disagree"], ["disagree as to"],
            ["issue", "dispute"], ["issue", "object", "whether"], ["point", "differ"], ["agree on which"]

        ],
        []
    ],
    "Identify the conclusion": [
        [
            ["expresses the conclusion"], ["expresses the overall conclusion"], ["the main conclusion", "is that"], ["expresses the argument's conclusion"],
            ["lead to which one of the following conclusions"], ["lead to which of the following conclusions"], ["states the conclusion"],
            ["is the main conclusion"], ["is the main point"], ["expresses the main point"], ["expresses the point of"], ["the main point", "is that"],
            ["the main point", "is to"], ["the basic position", "is that"], ["the point of", "is that"], ["the point", "expressed by"],
            ["express", "main conclusion"], ["which of the following can be", "concluded"], ["represent", "author's conclusion"]
        ],
        ["the author is arguing that", "what is the argument's conclusion"]
    ],
    "Identify the role": [
        [
            ["plays which one of the following roles"], ["serves which one of the following functions"],
            ["figures in", "in which one of the following ways"], ["used in", "in which one of the following ways"],
            ["plays which of the following roles"], ["serves which of the following functions"],
            ["figures in", "in which of the following ways"], ["used in", "in which of the following ways"], ["the clause", "serve", "as"],
            ["the statement", "offer", "in support of"], ["is offered", "as"], ["could be substituted for the reason"], ["describes the role played"],
        ],
        []
    ],
    "Parallel principle": [
        [
            ["the principle", "most similar"], ["the principle", "conforms to"], ["the principle", "most clearly violated"],
            ["the principle", "be justified by"], ["uses the principle"],
            ["the principle that underlies", "underlies which one of the following"], ["the principle that underlies", "underlies which of the following"],
            ["commits the fallacy described above"], ["this principle", "justify which"],
        ],
        []
    ],
    "Identify the principle": [
        [
            ["generalization", "illustrated by the passage"], ["provides the best illustration", "the principle"],
            ["which one of the following principles", "justify"], ["which one of the following principles", "justifies"],
            ["which one of the following principles", "provide", "justification"], ["which one of the following principles", "based on"],
            ["which one of the following principles", "support"],  ["which one of the following general principles", "support"],
            ["which one of the following principles", "provides a basis"], ["which one of the following principles", "would contribute"],
            ["which one of the following principles", "account for the contrast"], ["which one of the following principles", "would determine"],
            ["which one of the following principles", "provide", "backing"], ["which one of the following ethical criteria", "support"],
            ["illustrate", "which one of the following statements"], ["appeals to", "which one of the following principles"],
            ["illustrate", "which one of the following propositions"],
            ["which of the following principles", "justify"], ["which of the following principles", "justifies"],
            ["which of the following principles", "provide", "justification"], ["which of the following principles", "based on"],
            ["which of the following principles", "support"],  ["which of the following general principles", "support"],
            ["which of the following principles", "provides a basis"], ["which of the following principles", "would contribute"],
            ["which of the following principles", "account for the contrast"], ["which of the following principles", "would determine"],
            ["which of the following principles", "provide", "backing"], ["which of the following ethical criteria", "support"],
            ["illustrate", "which of the following statements"], ["appeals to", "which of the following principles"],
            ["illustrate", "which of the following propositions"],
            ["expresses a principle", "employed by the argument"],
            ["expresses a general principle"], ["if each is a principle"], ["a principle that,", "justifies"],
            ["serve as the principle", "appealed to"], ["conforms most closely to the"], ["most closely conform to"]
        ],
        ["most closely accords with which one of the following principles?", "conforms to which one of the following principles?",
         "conforms to which one of the following propositions?", "conforms to which one of the following generalizations?",
         "which one of the following is a principle", "expresses a general principle that could underlie the argument?"]
    ],
}


secondary_question_type_keyword_dict = {
    "Parallel flaw": [
        'flaw', 'questionable', 'erroneous', "faulty", "error", "mistake", "problem"
    ],
    "Identify the flaw": [
        'flaw', 'questionable', 'erroneous', 'criticism', "fail", "problem", "to mean that", "not succeed", "because", "points out", "mistake", "problem", "criticizing", "criticize"
    ],
    "Parallel reasoning": [
        'pattern', 'structure', "logical relationship", "logical structure"
    ],
    "Identify the technique": [ #***
        'technique', 'strategy', 'strategies', "function", 'proceeds by', 'by arguing that', 'seeks to establish that', 'utilized by the argument?',
        "the executive's reasoning does which", "method", "response to the", "logical relationship", "logical structure", "does which one of the following in",
        "is used to", "manner", "does which of the following in",
    ],
    "Parallel principle": [
        "principle", "generalization", "proposition", "policy", "example", "regulation", "policies"
    ],
    "Identify the principle": [
        "principle", "generalization", "proposition", "policy", "example", "policies"
    ],
    "Necessary assumption": [
        "necessary", "depend", "relies", "require", "assumes which one of the following", "assumes which of the following", "makes the assumption that", "assumption made by",
        "assumes that", "presupposition", "presuppose", "an assumption made", "have to be assumed that", "unless", "an assumption that the argument"
    ],
    "Sufficient assumption": [
        "sufficient", "which one of the following, if assumed", "if which one of the following is assumed?", "if which one of the following were assumed?",
        "which of the following, if assumed", "if which of the following is assumed?", "if which of the following were assumed?",  #***
    ],
    "Strengthen": [
        "strengthen", "improve", "contribute most to a defense", "most help to justify", "most helps to justify"
    ],
    "Weaken": [
        "weaken", 'undermine', 'weakness', 'counter', "objection", "argues most strongly", "most strongly indicates", "if true, indicates that",
        "calls into question", "strongest challenge to"
    ],
    "Evaluate": [
        "evaluate", "evaluating", "evaluation"
    ],
    "Paradox": [
        "explain", "resolve", "discrepancy", "explanation", "resolution", "reconcile", "contradictory"
    ],
    "Entailment": [
        'must be true', 'must also be true', 'follows logically', 'logically follows', 'can be inferred from',
        'can be properly inferred', 'could be true', 'count as evidence', 'provide a basis', 'justifiably be rejected',
        'provide reason for rejecting', "consequence", "conflicts with", " logically commit", "compatible", "have to be true",
        "evidence against", "consistent with", "above can best be understood as"
    ],
    "Most strongly supported": [
        'most strongly support', 'most support', 'best support', 'can be most reasonably inferred', 'most likely',
        'exhibit the most', 'most reasonably be concluded', "strongest support"
    ],
    "Complete the passage": [
        'complete', 'completion',
    ],
    "Dispute": [
        "dispute", "disagree", "agree that ", "issue", "agreeing about", "agree with each other that"
    ],
    "Identify the conclusion": [
        'main conclusion', 'main point', "overall conclusion"
    ],
    "Identify the role": [
        "role", "is to present"
    ],
}


def has_multiple_keywords(text, keywords_list):
    has_multiple_kw = False
    for keywords in keywords_list:
        has_cur_all_kw = True
        for each_kw in keywords:
            if each_kw not in text:
                has_cur_all_kw = False
                break

        if has_cur_all_kw is True:
            has_multiple_kw = True
            return has_multiple_kw
    return has_multiple_kw


def classify_question_type(question):
    question = question.lower()
    question_type = None

    if has_multiple_keywords(question, question_type_keyword_dict['Parallel flaw'][0]) or \
        has_keywords(question, question_type_keyword_dict['Parallel flaw'][1], is_bound=True):
        question_type = 'Parallel flaw'
    elif has_multiple_keywords(question, question_type_keyword_dict['Identify the flaw'][0]) or \
        has_keywords(question, question_type_keyword_dict['Identify the flaw'][1], is_bound=True):
        question_type = 'Identify the flaw'
    elif has_multiple_keywords(question, question_type_keyword_dict['Parallel reasoning'][0]) or \
        has_keywords(question, question_type_keyword_dict['Parallel reasoning'][1], is_bound=True):
        question_type = 'Parallel reasoning'
    elif has_multiple_keywords(question, question_type_keyword_dict['Identify the technique'][0]) or \
        has_keywords(question, question_type_keyword_dict['Identify the technique'][1], is_bound=True):
        question_type = 'Identify the technique'
    elif has_multiple_keywords(question, question_type_keyword_dict['Necessary assumption'][0]) or \
            has_keywords(question, question_type_keyword_dict['Necessary assumption'][1], is_bound=True):
        question_type = 'Necessary assumption'
    elif has_multiple_keywords(question, question_type_keyword_dict['Sufficient assumption'][0]) or \
            has_keywords(question, question_type_keyword_dict['Sufficient assumption'][1], is_bound=True):
        question_type = 'Sufficient assumption'
    elif has_multiple_keywords(question, question_type_keyword_dict['Weaken'][0]) or \
         has_keywords(question, question_type_keyword_dict['Weaken'][1], is_bound=True):
        question_type = 'Weaken'
    elif has_multiple_keywords(question, question_type_keyword_dict['Strengthen'][0]) or \
         has_keywords(question, question_type_keyword_dict['Strengthen'][1], is_bound=True):
        question_type = 'Strengthen'
    elif has_multiple_keywords(question, question_type_keyword_dict['Evaluate'][0]) or \
         has_keywords(question, question_type_keyword_dict['Evaluate'][1], is_bound=True):
        question_type = 'Evaluate'
    elif has_multiple_keywords(question, question_type_keyword_dict['Paradox'][0]) or \
            has_keywords(question, question_type_keyword_dict['Paradox'][1], is_bound=True):
        question_type = 'Paradox'
    elif has_multiple_keywords(question, question_type_keyword_dict['Entailment'][0]) or \
         has_keywords(question, question_type_keyword_dict['Entailment'][1], is_bound=True):
        question_type = 'Entailment'
    elif has_multiple_keywords(question, question_type_keyword_dict['Most strongly supported'][0]) or \
            has_keywords(question, question_type_keyword_dict['Most strongly supported'][1], is_bound=True):
        question_type = 'Most strongly supported'
    elif has_multiple_keywords(question, question_type_keyword_dict['Complete the passage'][0]) or \
         has_keywords(question, question_type_keyword_dict['Complete the passage'][1], is_bound=True):
        question_type = 'Complete the passage'
    elif has_multiple_keywords(question, question_type_keyword_dict['Dispute'][0]) or \
            has_keywords(question, question_type_keyword_dict['Dispute'][1], is_bound=True):
        question_type = 'Dispute'
    elif has_multiple_keywords(question, question_type_keyword_dict['Identify the conclusion'][0]) or \
         has_keywords(question, question_type_keyword_dict['Identify the conclusion'][1], is_bound=True):
        question_type = 'Identify the conclusion'
    elif has_multiple_keywords(question, question_type_keyword_dict['Identify the role'][0]) or \
         has_keywords(question, question_type_keyword_dict['Identify the role'][1], is_bound=True):
        question_type = 'Identify the role'
    elif has_multiple_keywords(question, question_type_keyword_dict['Parallel principle'][0]) or \
         has_keywords(question, question_type_keyword_dict['Parallel principle'][1], is_bound=True):
        question_type = 'Parallel principle'
    elif has_multiple_keywords(question, question_type_keyword_dict['Identify the principle'][0]) or \
         has_keywords(question, question_type_keyword_dict['Identify the principle'][1], is_bound=True):
        question_type = 'Identify the principle'

    return question_type


parallel_keywords = ["similar", "parallel", "resemble", "conform"]


# def select_question_type_from_multiple(type_list, is_parallel=True):
#     for each in type_list:
#         if is_parallel:
#             if "Parallel" in each:
#                 return each
#         else:
#             if "Parallel" not in each:
#                 return each
#     return type_list[0]

def select_question_type_from_multiple(type_list):
    for each in type_list:
        if "Parallel" in each:
            return each
    return type_list[0]


def extract_our_question_type(file_name):
    with open(file_name, "r") as f:
        lines = json.load(f)
        all_question_types = list()
        none_question_type_num = 0
        three_question_type_num = 0
        all_instance_num = 0

        for each_instance in lines:
            cur_question_type = classify_question_type(each_instance['question'])
            all_instance_num += 1
            if cur_question_type is None:
                candidate_question_types = list()
                for each_key in secondary_question_type_keyword_dict.keys():
                    if has_keywords(each_instance['question'].lower(), secondary_question_type_keyword_dict[each_key]):
                        candidate_question_types.append(each_key)
                if len(candidate_question_types) == 1:
                    all_question_types.append(candidate_question_types[0])
                elif len(candidate_question_types) == 0:
                    none_question_type_num += 1
                    all_question_types.append("")
                else:
                    if has_keywords(each_instance['question'], parallel_keywords):
                        all_question_types.append(select_question_type_from_multiple(candidate_question_types))
                    else:
                        all_question_types.append(candidate_question_types[0])
            else:
                all_question_types.append(cur_question_type)
        print(none_question_type_num, three_question_type_num, all_instance_num)
        return all_question_types


def edit_ques_type_description(ques_type_list):
    edited_ques_type_list = list()
    for each_type in ques_type_list:
        if each_type == "Entailment":
            edited_ques_type_list.append("Must be true or Cannot be true")
        elif each_type == "Dispute":
            edited_ques_type_list.append("Point at issue and disagreement")
        elif each_type == "Evaluate":
            edited_ques_type_list.append("Useful to know to evaluate")
        elif each_type == "Paradox":
            edited_ques_type_list.append("Explain or Resolve")
        else:
            edited_ques_type_list.append(each_type)
        # if each_type == "Identify the principle" or each_type == "Parallel principle":
        #     edited_ques_type_list.append("Principle")
        # elif each_type == "" or each_type == "Complete the passage":
        #     edited_ques_type_list.append("Others")
        # else:
        #     edited_ques_type_list.append(each_type)

    return edited_ques_type_list


inner_necessary_assumption_keywords = ["depend", "relies", "require", "necessary", "assumes which one of the following", "makes the assumption that", "assumption made by"]
bound_necessary_assumption_keywords = ["assumes that"]
inner_sufficient_assumption_keywords = ["follows logically", "logically follow", "properly drawn", "sufficient", "properly inferred", "if which one of the following is assumed?", "if which one of the following were assumed?"]
inner_technique_keywords = ['strategies', " responds to ", " challenges ", "function"]
bound_technique_keywords = ['proceeds by', ' by', 'by arguing that', 'seeks to establish that', 'utilized by the argument?', "The executive's reasoning does which"]
inner_role_keywords = ['role', ]
bound_role_keywords = ['The claim that', "The clause"]
inner_entailment_keywords = ['must be true', 'must also be true', 'follows logically', 'logically follows', 'can be inferred from', 'can be properly inferred', 'could be true', 'count as evidence', 'provide a basis', 'justifiably be rejected', 'provide reason for rejecting']
inner_most_supported_keywords = ['most strongly support', 'most support', 'best support', 'can be most reasonably inferred', 'most likely', 'exhibit the most', 'most reasonably be concluded']
inner_complete_keywords = ['most logically completes', 'most logical completion', 'most reasonably completes']

identify_flaw_keywords = ['criticism', 'flaw']
parallel_flaw_keywords = ['flaw', 'questionable', 'erroneous']
parallel_reasoning_keywords = ['similar', 'parallel']
paradox_keywords = ['explain', 'resolve']
weaken_keywords = ['calls into', 'weaken', 'undermine', 'weakness', 'counter']
strengthen_keywords = ['strengthen', 'justifies']
evaluate_keywords = ['evaluate', 'evaluating']
conclusion_keywords = ['main conclusion', 'main point', 'conclusion', 'The author is arguing that']
dispute_keywords = ['disagree', 'dispute', 'agree that']
identify_principle_keywords = ['principle']
parallel_principle_keywords = ['principle']

def has_keywords(text, keywords, is_bound=False):
    has_kw = False
    for each_kw in keywords:
        if not is_bound:
            if each_kw in text:
                has_kw = True
                return has_kw
        else:
            if (len(text) - len(each_kw) >=0 and text.find(each_kw) == len(text) - len(each_kw)) or text.find(each_kw) == 0:
                has_kw = True
                return has_kw
    return has_kw



def calculate_question_type_distribution(all_question_types):
    all_question_type_nums = [0, ] * len(question_type_list)
    for each_type in all_question_types:
        ques_type_index = question_type_list.index(each_type)
        all_question_type_nums[ques_type_index] += 1

    all_question_type_ratios = [np.round(each_num/len(all_question_types), 4) for each_num in all_question_type_nums]
    print(all_question_type_ratios, sum(all_question_type_ratios))


GT_ques_type_names = ["Necessary assumption", "Sufficient assumption", "Strengthen", "Weaken", "Evaluate",
                      "Entailment", "Identify the conclusion", "Most strongly supported", "Paradox",
                      "Principle", "Dispute", "Identify the technique", "Identify the role", "Identify the flaw", "Parallel flaw",
                      "Parallel reasoning", "Others"]
def compare_test_GT_question_type(file_name, our_extracted_types):
    with open(file_name, "r") as f:
        lines = json.load(f)
        all_question_types = list()
        different_num = 0

        for i, each_instance in enumerate(lines):
            cur_question_type = GT_ques_type_names[each_instance['question_type']]
            all_question_types.append(cur_question_type)

            if cur_question_type != our_extracted_types[i]: # and cur_question_type != "Others":
                different_num += 1
                if different_num > 40 and different_num < 60:
                    print(cur_question_type, "\t", our_extracted_types[i])
                    print(each_instance['question'])
                    print("*"*100)
        print(different_num, len(our_extracted_types), different_num/len(our_extracted_types))


if __name__ == "__main__":
    lr_train_file = '../reclor-data/train.json'
    lr_val_file = '../reclor-data/val.json'
    lr_test_file = '../reclor-data/test.json'

    # calculate_question_type_distribution(all_question_types_test)

    # get question type of each question
    all_question_types_train_our = extract_our_question_type(lr_train_file)
    all_question_types_val_our = extract_our_question_type(lr_val_file)
    all_question_types_test_our = extract_our_question_type(lr_test_file)

    # enrich the description of each question type
    all_question_types_train_our = edit_ques_type_description(all_question_types_train_our)
    all_question_types_val_our = edit_ques_type_description(all_question_types_val_our)
    all_question_types_test_our = edit_ques_type_description(all_question_types_test_our)

    # save question type
    ques_type_train_file = 'reclor-data/train_ques_types'
    ques_type_val_file = 'reclor-data/val_ques_types'
    ques_type_test_file = 'reclor-data/test_ques_types'
    np.save(ques_type_train_file, all_question_types_train_our)
    np.save(ques_type_val_file, all_question_types_val_our)
    np.save(ques_type_test_file, all_question_types_test_our)

    # compare_test_GT_question_type(lr_test_file, all_question_types_test_our)



