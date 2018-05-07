
def calc_pos_protected_percents(predicted, sensitive, unprotected_vals, positive_pred):
    """
    Returns P(C=YES|sensitive=privileged) and P(C=YES|sensitive=not privileged) in that order where
    C is the predicited classification and where all not privileged values are considered
    equivalent.  Assumes that predicted and sensitive have the same lengths.
    """
    unprotected_positive = 0.0
    unprotected_negative = 0.0
    protected_positive = 0.0
    protected_negative = 0.0
    for i in range(0, len(predicted)):
        protected_val = sensitive[i]
        predicted_val = predicted[i]
        if protected_val in unprotected_vals:
            if str(predicted_val) == str(positive_pred):
                unprotected_positive += 1
            else:
                unprotected_negative += 1
        else:
            if str(predicted_val) == str(positive_pred):
                protected_positive += 1
            else:
                protected_negative += 1

    protected_pos_percent = 0.0
    if protected_positive + protected_negative > 0:
        protected_pos_percent = protected_positive / (protected_positive + protected_negative)
    unprotected_pos_percent = 0.0
    if unprotected_positive + unprotected_negative > 0:
        unprotected_pos_percent = unprotected_positive /  \
                                  (unprotected_positive + unprotected_negative)

    return unprotected_pos_percent, protected_pos_percent


def calc_prob_class_given_sensitive(predicted, sensitive, predicted_goal, sensitive_goal):
    """
    Returns P(predicted = predicted_goal | sensitive = sensitive_goal).  Assumes that predicted
    and sensitive have the same length.  If there are no attributes matching the given
    sensitive_goal, this will error.
    """
    match_count = 0.0
    total = 0.0
    for sens, pred in zip(sensitive, predicted):
        if str(sens) == str(sensitive_goal):
            total += 1
            if str(pred) == str(predicted_goal):
                match_count += 1

    return match_count / total

def calc_fp_fn(actual, predicted, sensitive, unprotected_vals, positive_pred):
    """
    Returns False positive and false negative for protected and unprotected group.
    """
    unprotected_negative = 0.0
    protected_positive = 0.0
    protected_negative = 0.0
    fp_protected = 0.0
    fp_unprotected = 0.0
    fn_protected=0.0
    fn_unprotected=0.0
    fp_diff =0.0
    for i in range(0, len(predicted)):
        protected_val = sensitive[i]
        predicted_val = predicted[i]
        actual_val= actual[i]
        if protected_val in unprotected_vals:
            if (str(predicted_val)==str(positive_pred))&(str(actual_val)!=str(predicted_val)):
                fp_unprotected+=1
            elif(str(predicted_val)!=str(positive_pred))&(str(actual_val)==str(predicted_val)):
                fn_unprotected+=1
        else:
            if (str(predicted_val)==str(positive_pred))&(str(actual_val)!=str(predicted_val)):
                    fp_protected+=1
            elif(str(predicted_val)!=str(positive_pred))&(str(actual_val)==str(predicted_val)):
                    fn_protected+=1
    return fp_unprotected,fp_protected, fn_protected, fn_unprotected

