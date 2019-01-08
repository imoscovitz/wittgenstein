class Metrics:
    def __init__(self):
        self._ = '_'
        
    def gain(self, other, pos_df, neg_df):
        """ Returns the information gain from self to other """

        p0count = self.num_covered(pos_df)
        p1count = other.num_covered(pos_df)
        n0count = self.num_covered(neg_df)
        n1count = other.num_covered(neg_df)
        return p1count * (math.log2((p1count + 1) / (p1count + n1count + 1)) - math.log2((p0count + 1) / (p0count + n0count + 1)))

    def precision(object, pos_df, neg_df):
        """ Returns precision value of object's classification.
            object: Cond, Rule, or Ruleset
        """

        pos_covered = object.covers(pos_df)
        neg_covered = object.covers(neg_df)
        total_n_covered = len(pos_covered)+len(neg_covered)
        if total_n_covered == 0:
            return None
        else:
            return len(pos_covered) / total_n_covered

    def score_accuracy(predictions, actuals):
        """ For evaluating trained model on test set.

            predictions: <iterable<bool>> True for predicted positive class, False otherwise
            actuals:     <iterable<bool>> True for actual positive class, False otherwise
        """
        t = [pr for pr,act in zip(predictions,actuals) if pr==act]
        n = predictions
        print(f't {t}')
        print(f't {n}')
        return len(t)/len(n)

    def accuracy(object, pos_pruneset, neg_pruneset):
        """ Returns accuracy value of object's classification.
            object: Cond, Rule, or Ruleset
        """
        P = len(pos_pruneset)
        N = len(neg_pruneset)
        if P + N == 0:
            return None

        tp = len(object.covers(pos_pruneset))
        tn = N - len(object.covers(neg_pruneset))
        return (tp + tn) / (P + N)
