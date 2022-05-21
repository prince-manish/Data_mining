import scipy.stats as stats

categories = [[250, 200], [50, 1000]]
chi2, p, dof, exp = stats.chi2_contingency(categories)
# input
#                      male           female        sum(row)
#  fiction             250              200            450
#  Non - fiction        50              1000           1050
#  sum(Col.)           300              1200           1500
print("Cal chi square = ", chi2, ", table value = ", p, ", degree of freedom = ", dof, ", Expected value = ", exp)
alpha = 0.001
if p <= alpha:
    print("we reject the null hypothesis")
else:
    print("Null hypothesis can not rejected")
