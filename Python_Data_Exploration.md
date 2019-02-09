
# Python Data Exploration

"""
Python for Data Science Study - Exploration

Created on Thu Feb  7 19:14:08 2019

Theme from: [dunovank](https://github.com/dunovank/iPython-Notebook-Theme)

@author: baocai
"""


```python
# Set up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df_loan = pd.read_csv('LoanStats3a.csv', skiprows = 0, header = 1, low_memory = False)
```

The data is filled with nulls

Some of the columns are completely empty, we can take those out


```python
df_loan.info(verbose = True, null_counts = True)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 42538 entries, 0 to 42537
    Data columns (total 145 columns):
    id                                            3 non-null object
    member_id                                     0 non-null float64
    loan_amnt                                     42535 non-null float64
    funded_amnt                                   42535 non-null float64
    funded_amnt_inv                               42535 non-null float64
    term                                          42535 non-null object
    int_rate                                      42535 non-null object
    installment                                   42535 non-null float64
    grade                                         42535 non-null object
    sub_grade                                     42535 non-null object
    emp_title                                     39909 non-null object
    emp_length                                    41423 non-null object
    home_ownership                                42535 non-null object
    annual_inc                                    42531 non-null float64
    verification_status                           42535 non-null object
    issue_d                                       42535 non-null object
    loan_status                                   42535 non-null object
    pymnt_plan                                    42535 non-null object
    url                                           0 non-null float64
    desc                                          29242 non-null object
    purpose                                       42535 non-null object
    title                                         42522 non-null object
    zip_code                                      42535 non-null object
    addr_state                                    42535 non-null object
    dti                                           42535 non-null float64
    delinq_2yrs                                   42506 non-null float64
    earliest_cr_line                              42506 non-null object
    inq_last_6mths                                42506 non-null float64
    mths_since_last_delinq                        15609 non-null float64
    mths_since_last_record                        3651 non-null float64
    open_acc                                      42506 non-null float64
    pub_rec                                       42506 non-null float64
    revol_bal                                     42535 non-null float64
    revol_util                                    42445 non-null object
    total_acc                                     42506 non-null float64
    initial_list_status                           42535 non-null object
    out_prncp                                     42535 non-null float64
    out_prncp_inv                                 42535 non-null float64
    total_pymnt                                   42535 non-null float64
    total_pymnt_inv                               42535 non-null float64
    total_rec_prncp                               42535 non-null float64
    total_rec_int                                 42535 non-null float64
    total_rec_late_fee                            42535 non-null float64
    recoveries                                    42535 non-null float64
    collection_recovery_fee                       42535 non-null float64
    last_pymnt_d                                  42452 non-null object
    last_pymnt_amnt                               42535 non-null float64
    next_pymnt_d                                  2749 non-null object
    last_credit_pull_d                            42531 non-null object
    collections_12_mths_ex_med                    42390 non-null float64
    mths_since_last_major_derog                   0 non-null float64
    policy_code                                   42535 non-null float64
    application_type                              42535 non-null object
    annual_inc_joint                              0 non-null float64
    dti_joint                                     0 non-null float64
    verification_status_joint                     0 non-null float64
    acc_now_delinq                                42506 non-null float64
    tot_coll_amt                                  0 non-null float64
    tot_cur_bal                                   0 non-null float64
    open_acc_6m                                   0 non-null float64
    open_act_il                                   0 non-null float64
    open_il_12m                                   0 non-null float64
    open_il_24m                                   0 non-null float64
    mths_since_rcnt_il                            0 non-null float64
    total_bal_il                                  0 non-null float64
    il_util                                       0 non-null float64
    open_rv_12m                                   0 non-null float64
    open_rv_24m                                   0 non-null float64
    max_bal_bc                                    0 non-null float64
    all_util                                      0 non-null float64
    total_rev_hi_lim                              0 non-null float64
    inq_fi                                        0 non-null float64
    total_cu_tl                                   0 non-null float64
    inq_last_12m                                  0 non-null float64
    acc_open_past_24mths                          0 non-null float64
    avg_cur_bal                                   0 non-null float64
    bc_open_to_buy                                0 non-null float64
    bc_util                                       0 non-null float64
    chargeoff_within_12_mths                      42390 non-null float64
    delinq_amnt                                   42506 non-null float64
    mo_sin_old_il_acct                            0 non-null float64
    mo_sin_old_rev_tl_op                          0 non-null float64
    mo_sin_rcnt_rev_tl_op                         0 non-null float64
    mo_sin_rcnt_tl                                0 non-null float64
    mort_acc                                      0 non-null float64
    mths_since_recent_bc                          0 non-null float64
    mths_since_recent_bc_dlq                      0 non-null float64
    mths_since_recent_inq                         0 non-null float64
    mths_since_recent_revol_delinq                0 non-null float64
    num_accts_ever_120_pd                         0 non-null float64
    num_actv_bc_tl                                0 non-null float64
    num_actv_rev_tl                               0 non-null float64
    num_bc_sats                                   0 non-null float64
    num_bc_tl                                     0 non-null float64
    num_il_tl                                     0 non-null float64
    num_op_rev_tl                                 0 non-null float64
    num_rev_accts                                 0 non-null float64
    num_rev_tl_bal_gt_0                           0 non-null float64
    num_sats                                      0 non-null float64
    num_tl_120dpd_2m                              0 non-null float64
    num_tl_30dpd                                  0 non-null float64
    num_tl_90g_dpd_24m                            0 non-null float64
    num_tl_op_past_12m                            0 non-null float64
    pct_tl_nvr_dlq                                0 non-null float64
    percent_bc_gt_75                              0 non-null float64
    pub_rec_bankruptcies                          41170 non-null float64
    tax_liens                                     42430 non-null float64
    tot_hi_cred_lim                               0 non-null float64
    total_bal_ex_mort                             0 non-null float64
    total_bc_limit                                0 non-null float64
    total_il_high_credit_limit                    0 non-null float64
    revol_bal_joint                               0 non-null float64
    sec_app_earliest_cr_line                      0 non-null float64
    sec_app_inq_last_6mths                        0 non-null float64
    sec_app_mort_acc                              0 non-null float64
    sec_app_open_acc                              0 non-null float64
    sec_app_revol_util                            0 non-null float64
    sec_app_open_act_il                           0 non-null float64
    sec_app_num_rev_accts                         0 non-null float64
    sec_app_chargeoff_within_12_mths              0 non-null float64
    sec_app_collections_12_mths_ex_med            0 non-null float64
    sec_app_mths_since_last_major_derog           0 non-null float64
    hardship_flag                                 42535 non-null object
    hardship_type                                 0 non-null float64
    hardship_reason                               0 non-null float64
    hardship_status                               0 non-null float64
    deferral_term                                 0 non-null float64
    hardship_amount                               0 non-null float64
    hardship_start_date                           0 non-null float64
    hardship_end_date                             0 non-null float64
    payment_plan_start_date                       0 non-null float64
    hardship_length                               0 non-null float64
    hardship_dpd                                  0 non-null float64
    hardship_loan_status                          0 non-null float64
    orig_projected_additional_accrued_interest    0 non-null float64
    hardship_payoff_balance_amount                0 non-null float64
    hardship_last_payment_amount                  0 non-null float64
    disbursement_method                           42535 non-null object
    debt_settlement_flag                          42535 non-null object
    debt_settlement_flag_date                     160 non-null object
    settlement_status                             160 non-null object
    settlement_date                               160 non-null object
    settlement_amount                             160 non-null float64
    settlement_percentage                         160 non-null float64
    settlement_term                               160 non-null float64
    dtypes: float64(115), object(30)
    memory usage: 47.1+ MB



```python
df_loan.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>...</th>
      <th>hardship_payoff_balance_amount</th>
      <th>hardship_last_payment_amount</th>
      <th>disbursement_method</th>
      <th>debt_settlement_flag</th>
      <th>debt_settlement_flag_date</th>
      <th>settlement_status</th>
      <th>settlement_date</th>
      <th>settlement_amount</th>
      <th>settlement_percentage</th>
      <th>settlement_term</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>4975.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>60 months</td>
      <td>12.69%</td>
      <td>67.79</td>
      <td>B</td>
      <td>B5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 145 columns</p>
</div>




```python
# There are too many columns that we cannot use because the data is missing
# We cannot download the fully-filled dataset without an account
loan_rows = len(df_loan)
df_dropped = pd.DataFrame()
for col in df_loan.columns:
    if df_loan[col].isnull().sum() >= 0.9*loan_rows:
#         df_dropped[col] = df_loan.pop(col) # In case we need the dropped columns
        df_loan.pop(col)
```


```python
# Now it looks more like a useful dataset
df_loan.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 42538 entries, 0 to 42537
    Data columns (total 55 columns):
    loan_amnt                     42535 non-null float64
    funded_amnt                   42535 non-null float64
    funded_amnt_inv               42535 non-null float64
    term                          42535 non-null object
    int_rate                      42535 non-null object
    installment                   42535 non-null float64
    grade                         42535 non-null object
    sub_grade                     42535 non-null object
    emp_title                     39909 non-null object
    emp_length                    41423 non-null object
    home_ownership                42535 non-null object
    annual_inc                    42531 non-null float64
    verification_status           42535 non-null object
    issue_d                       42535 non-null object
    loan_status                   42535 non-null object
    pymnt_plan                    42535 non-null object
    desc                          29242 non-null object
    purpose                       42535 non-null object
    title                         42522 non-null object
    zip_code                      42535 non-null object
    addr_state                    42535 non-null object
    dti                           42535 non-null float64
    delinq_2yrs                   42506 non-null float64
    earliest_cr_line              42506 non-null object
    inq_last_6mths                42506 non-null float64
    mths_since_last_delinq        15609 non-null float64
    open_acc                      42506 non-null float64
    pub_rec                       42506 non-null float64
    revol_bal                     42535 non-null float64
    revol_util                    42445 non-null object
    total_acc                     42506 non-null float64
    initial_list_status           42535 non-null object
    out_prncp                     42535 non-null float64
    out_prncp_inv                 42535 non-null float64
    total_pymnt                   42535 non-null float64
    total_pymnt_inv               42535 non-null float64
    total_rec_prncp               42535 non-null float64
    total_rec_int                 42535 non-null float64
    total_rec_late_fee            42535 non-null float64
    recoveries                    42535 non-null float64
    collection_recovery_fee       42535 non-null float64
    last_pymnt_d                  42452 non-null object
    last_pymnt_amnt               42535 non-null float64
    last_credit_pull_d            42531 non-null object
    collections_12_mths_ex_med    42390 non-null float64
    policy_code                   42535 non-null float64
    application_type              42535 non-null object
    acc_now_delinq                42506 non-null float64
    chargeoff_within_12_mths      42390 non-null float64
    delinq_amnt                   42506 non-null float64
    pub_rec_bankruptcies          41170 non-null float64
    tax_liens                     42430 non-null float64
    hardship_flag                 42535 non-null object
    disbursement_method           42535 non-null object
    debt_settlement_flag          42535 non-null object
    dtypes: float64(30), object(25)
    memory usage: 17.8+ MB



```python
# Next is to make sense of what we are looking at
df_loan.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>...</th>
      <th>policy_code</th>
      <th>application_type</th>
      <th>acc_now_delinq</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_amnt</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
      <th>hardship_flag</th>
      <th>disbursement_method</th>
      <th>debt_settlement_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>4975.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>...</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>Ryder</td>
      <td>&lt; 1 year</td>
      <td>...</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>...</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>AIR RESOURCES BOARD</td>
      <td>10+ years</td>
      <td>...</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>60 months</td>
      <td>12.69%</td>
      <td>67.79</td>
      <td>B</td>
      <td>B5</td>
      <td>University Medical Group</td>
      <td>1 year</td>
      <td>...</td>
      <td>1.0</td>
      <td>Individual</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>N</td>
      <td>Cash</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 55 columns</p>
</div>




```python
# Luckily we have a dictionary file from the source
df_dictionary = pd.read_csv('DataDictionary.csv')
# We just need the description column
df_dictionary = df_dictionary.iloc[:,0:2]
# Set this to join with the columns we are interested in
df_dictionary = df_dictionary.set_index('LoanStatNew')
```


```python
pd.set_option('display.max_colwidth', -1)
df_summary = pd.DataFrame(data = df_loan.columns, columns = ['Column']).join(df_dictionary, on = ['Column'], how = 'inner')
```


```python
df_summary.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>loan_amnt</td>
      <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>funded_amnt</td>
      <td>The total amount committed to that loan at that point in time.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>funded_amnt_inv</td>
      <td>The total amount committed by investors for that loan at that point in time.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>term</td>
      <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>int_rate</td>
      <td>Interest Rate on the loan</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a summary table to see the descriptions, types and some values for reference
df_summary['Type'] = None
df_summary['Examples'] = None

for i in range(len(df_summary['Type'])):
    df_summary['Type'][i] = df_loan.iloc[:, i].dtype
    df_summary['Examples'][i] = pd.unique(df_loan.iloc[:, i])[:6]
```

There are some variables that are worth noticing

>index #6, #7, #10, #12: Can be left as is or categorize<br>
>index #0, #1, #2, #5, #11, #21, #22, #24 - #28, #30, #34 - #40: Are continuous variables<br>
>index #4, #29: Values in percentages, the type should be float<br>
>index #13, #23, #41, #43: Are Months and Years<br>
>index #3 - term: The column is object while it can be integer, further categorical<br>
>index #8 - emp-title: Might be useful, might be not, just unique strings<br>
>index #9 - emp-length: Description indicated integer categorical data, while actual said object<br>
>index #13 - issue_d: can be separated into Month and Year columns<br>
>index #14 - loan_status: certainly can be categorized but seems not standardized, need further inspection<br>
>index #15, #32, #33: only 2 unique values, one of them is NA, could be categorical, otherwise can drop this<br>
>index #16 - desc: might contain useful information, but not without contexts -> drop<br>
>index #17 & #18: #18 is a longer version of #17<br>
>index #19 & #20: instead of using zipcode, addr_state will be used for ease of viewing<br>
>index #23: seems like a history of borrowing, can be combined with #13 for analysis<br>
>index #31: 2 possible values, should be categorical, but only 1 exists in the dataset<br>
>index #34 & #35: seems similar, can check then drop either one<br>
>index #44 - #53: missing categorical values, represented by NA

Normally, the NA values here should be double checked with the source of data before assuming/changing for analysis. Here, that option is not on the table, so those with too much NA will be ommitted.


```python
df_summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Description</th>
      <th>Type</th>
      <th>Examples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>loan_amnt</td>
      <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
      <td>float64</td>
      <td>[5000.0, 2500.0, 2400.0, 10000.0, 3000.0, 7000.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>funded_amnt</td>
      <td>The total amount committed to that loan at that point in time.</td>
      <td>float64</td>
      <td>[5000.0, 2500.0, 2400.0, 10000.0, 3000.0, 7000.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>funded_amnt_inv</td>
      <td>The total amount committed by investors for that loan at that point in time.</td>
      <td>float64</td>
      <td>[4975.0, 2500.0, 2400.0, 10000.0, 3000.0, 5000.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>term</td>
      <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
      <td>object</td>
      <td>[ 36 months,  60 months, nan]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>int_rate</td>
      <td>Interest Rate on the loan</td>
      <td>object</td>
      <td>[ 10.65%,  15.27%,  15.96%,  13.49%,  12.69%,   7.90%]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>installment</td>
      <td>The monthly payment owed by the borrower if the loan originates.</td>
      <td>float64</td>
      <td>[162.87, 59.83, 84.33, 339.31, 67.79, 156.46]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>grade</td>
      <td>LC assigned loan grade</td>
      <td>object</td>
      <td>[B, C, A, E, F, D]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sub_grade</td>
      <td>LC assigned loan subgrade</td>
      <td>object</td>
      <td>[B2, C4, C5, C1, B5, A4]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>emp_title</td>
      <td>The job title supplied by the Borrower when applying for the loan.*</td>
      <td>object</td>
      <td>[nan, Ryder, AIR RESOURCES BOARD, University Medical Group, Veolia Transportaton, Southern Star Photography]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>emp_length</td>
      <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
      <td>object</td>
      <td>[10+ years, &lt; 1 year, 1 year, 3 years, 8 years, 9 years]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>home_ownership</td>
      <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
      <td>object</td>
      <td>[RENT, OWN, MORTGAGE, OTHER, NONE, nan]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>annual_inc</td>
      <td>The self-reported annual income provided by the borrower during registration.</td>
      <td>float64</td>
      <td>[24000.0, 30000.0, 12252.0, 49200.0, 80000.0, 36000.0]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>verification_status</td>
      <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
      <td>object</td>
      <td>[Verified, Source Verified, Not Verified, nan]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>issue_d</td>
      <td>The month which the loan was funded</td>
      <td>object</td>
      <td>[Dec-2011, Nov-2011, Oct-2011, Sep-2011, Aug-2011, Jul-2011]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>loan_status</td>
      <td>Current status of the loan</td>
      <td>object</td>
      <td>[Fully Paid, Charged Off, nan, Does not meet the credit policy. Status:Fully Paid, Does not meet the credit policy. Status:Charged Off]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>pymnt_plan</td>
      <td>Indicates if a payment plan has been put in place for the loan</td>
      <td>object</td>
      <td>[n, nan]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>desc</td>
      <td>Loan description provided by the borrower</td>
      <td>object</td>
      <td>[  Borrower added on 12/22/11 &gt; I need to upgrade my business technologies.&lt;br&gt;,   Borrower added on 12/22/11 &gt; I plan to use this money to finance the motorcycle i am looking at. I plan to have it paid off as soon as possible/when i sell my old bike. I only need this money because the deal im looking at is to good to pass up.&lt;br&gt;&lt;br&gt;  Borrower added on 12/22/11 &gt; I plan to use this money to finance the motorcycle i am looking at. I plan to have it paid off as soon as possible/when i sell my old bike.I only need this money because the deal im looking at is to good to pass up. I have finished college with an associates degree in business and its takingmeplaces&lt;br&gt;, nan,   Borrower added on 12/21/11 &gt; to pay for property tax (borrow from friend, need to pay back) &amp; central A/C need to be replace. I'm very sorry to let my loan expired last time.&lt;br&gt;,   Borrower added on 12/21/11 &gt; I plan on combining three large interest bills together and freeing up some extra each month to pay toward other bills.  I've always been a good payor but have found myself needing to make adjustments to my budget due to a medical scare. My job is very stable, I love it.&lt;br&gt;,   Borrower added on 12/18/11 &gt; I am planning on using the funds to pay off two retail credit cards with 24.99% interest rates, as well as a major bank credit card with a 18.99% rate.  I pay all my bills on time, looking for a lower combined payment and lower monthly payment.&lt;br&gt;]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>purpose</td>
      <td>A category provided by the borrower for the loan request.</td>
      <td>object</td>
      <td>[credit_card, car, small_business, other, wedding, debt_consolidation]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>title</td>
      <td>The loan title provided by the borrower</td>
      <td>object</td>
      <td>[Computer, bike, real estate business, personel, Personal, My wedding loan I promise to pay back]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>zip_code</td>
      <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
      <td>object</td>
      <td>[860xx, 309xx, 606xx, 917xx, 972xx, 852xx]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>addr_state</td>
      <td>The state provided by the borrower in the loan application</td>
      <td>object</td>
      <td>[AZ, GA, IL, CA, OR, NC]</td>
    </tr>
    <tr>
      <th>21</th>
      <td>dti</td>
      <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
      <td>float64</td>
      <td>[27.65, 1.0, 8.72, 20.0, 17.94, 11.2]</td>
    </tr>
    <tr>
      <th>22</th>
      <td>delinq_2yrs</td>
      <td>The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years</td>
      <td>float64</td>
      <td>[0.0, 2.0, 3.0, 1.0, 4.0, 6.0]</td>
    </tr>
    <tr>
      <th>23</th>
      <td>earliest_cr_line</td>
      <td>The month the borrower's earliest reported credit line was opened</td>
      <td>object</td>
      <td>[Jan-1985, Apr-1999, Nov-2001, Feb-1996, Jan-1996, Nov-2004]</td>
    </tr>
    <tr>
      <th>24</th>
      <td>inq_last_6mths</td>
      <td>The number of inquiries in past 6 months (excluding auto and mortgage inquiries)</td>
      <td>float64</td>
      <td>[1.0, 5.0, 2.0, 0.0, 3.0, 4.0]</td>
    </tr>
    <tr>
      <th>25</th>
      <td>mths_since_last_delinq</td>
      <td>The number of months since the borrower's last delinquency.</td>
      <td>float64</td>
      <td>[nan, 35.0, 38.0, 61.0, 8.0, 20.0]</td>
    </tr>
    <tr>
      <th>26</th>
      <td>open_acc</td>
      <td>The number of open credit lines in the borrower's credit file.</td>
      <td>float64</td>
      <td>[3.0, 2.0, 10.0, 15.0, 9.0, 7.0]</td>
    </tr>
    <tr>
      <th>27</th>
      <td>pub_rec</td>
      <td>Number of derogatory public records</td>
      <td>float64</td>
      <td>[0.0, 1.0, 2.0, 3.0, 4.0, nan]</td>
    </tr>
    <tr>
      <th>28</th>
      <td>revol_bal</td>
      <td>Total credit revolving balance</td>
      <td>float64</td>
      <td>[13648.0, 1687.0, 2956.0, 5598.0, 27783.0, 7963.0]</td>
    </tr>
    <tr>
      <th>29</th>
      <td>revol_util</td>
      <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
      <td>object</td>
      <td>[83.7%, 9.4%, 98.5%, 21%, 53.9%, 28.3%]</td>
    </tr>
    <tr>
      <th>30</th>
      <td>total_acc</td>
      <td>The total number of credit lines currently in the borrower's credit file</td>
      <td>float64</td>
      <td>[9.0, 4.0, 10.0, 37.0, 38.0, 12.0]</td>
    </tr>
    <tr>
      <th>31</th>
      <td>initial_list_status</td>
      <td>The initial listing status of the loan. Possible values are – W, F</td>
      <td>object</td>
      <td>[f, nan]</td>
    </tr>
    <tr>
      <th>32</th>
      <td>out_prncp</td>
      <td>Remaining outstanding principal for total amount funded</td>
      <td>float64</td>
      <td>[0.0, nan]</td>
    </tr>
    <tr>
      <th>33</th>
      <td>out_prncp_inv</td>
      <td>Remaining outstanding principal for portion of total amount funded by investors</td>
      <td>float64</td>
      <td>[0.0, nan]</td>
    </tr>
    <tr>
      <th>34</th>
      <td>total_pymnt</td>
      <td>Payments received to date for total amount funded</td>
      <td>float64</td>
      <td>[5863.1551866952, 1014.53, 3005.6668441393, 12231.890000000902, 4066.9081610816997, 5632.209999999401]</td>
    </tr>
    <tr>
      <th>35</th>
      <td>total_pymnt_inv</td>
      <td>Payments received to date for portion of total amount funded by investors</td>
      <td>float64</td>
      <td>[5833.84, 1014.53, 3005.67, 12231.89, 4066.91, 5632.21]</td>
    </tr>
    <tr>
      <th>36</th>
      <td>total_rec_prncp</td>
      <td>Principal received to date</td>
      <td>float64</td>
      <td>[5000.0, 456.46, 2400.0, 10000.0, 3000.0, 7000.0]</td>
    </tr>
    <tr>
      <th>37</th>
      <td>total_rec_int</td>
      <td>Interest received to date</td>
      <td>float64</td>
      <td>[863.16, 435.17, 605.67, 2214.92, 1066.91, 632.21]</td>
    </tr>
    <tr>
      <th>38</th>
      <td>total_rec_late_fee</td>
      <td>Late fees received to date</td>
      <td>float64</td>
      <td>[0.0, 16.97, 15.000000030499999, 24.17, 15.000000000014, 1.0]</td>
    </tr>
    <tr>
      <th>39</th>
      <td>recoveries</td>
      <td>post charge off gross recovery</td>
      <td>float64</td>
      <td>[0.0, 122.9, 190.54, 277.69, 450.92, 645.1]</td>
    </tr>
    <tr>
      <th>40</th>
      <td>collection_recovery_fee</td>
      <td>post charge off collection fee</td>
      <td>float64</td>
      <td>[0.0, 1.11, 2.09, 2.52, 4.16, 6.3145]</td>
    </tr>
    <tr>
      <th>41</th>
      <td>last_pymnt_d</td>
      <td>Last month payment was received</td>
      <td>object</td>
      <td>[Jan-2015, Apr-2013, Jun-2014, Jan-2017, May-2016, Apr-2012]</td>
    </tr>
    <tr>
      <th>42</th>
      <td>last_pymnt_amnt</td>
      <td>Last total payment amount received</td>
      <td>float64</td>
      <td>[171.62, 119.66, 649.91, 357.48, 67.3, 161.03]</td>
    </tr>
    <tr>
      <th>43</th>
      <td>last_credit_pull_d</td>
      <td>The most recent month LC pulled credit for this loan</td>
      <td>object</td>
      <td>[Dec-2018, Oct-2016, Jun-2017, Apr-2016, Apr-2018, Feb-2017]</td>
    </tr>
    <tr>
      <th>44</th>
      <td>collections_12_mths_ex_med</td>
      <td>Number of collections in 12 months excluding medical collections</td>
      <td>float64</td>
      <td>[0.0, nan]</td>
    </tr>
    <tr>
      <th>45</th>
      <td>policy_code</td>
      <td>publicly available policy_code=1\nnew products not publicly available policy_code=2</td>
      <td>float64</td>
      <td>[1.0, nan]</td>
    </tr>
    <tr>
      <th>46</th>
      <td>application_type</td>
      <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
      <td>object</td>
      <td>[Individual, nan]</td>
    </tr>
    <tr>
      <th>47</th>
      <td>acc_now_delinq</td>
      <td>The number of accounts on which the borrower is now delinquent.</td>
      <td>float64</td>
      <td>[0.0, nan, 1.0]</td>
    </tr>
    <tr>
      <th>48</th>
      <td>chargeoff_within_12_mths</td>
      <td>Number of charge-offs within 12 months</td>
      <td>float64</td>
      <td>[0.0, nan]</td>
    </tr>
    <tr>
      <th>49</th>
      <td>delinq_amnt</td>
      <td>The past-due amount owed for the accounts on which the borrower is now delinquent.</td>
      <td>float64</td>
      <td>[0.0, nan, 27.0, 6053.0]</td>
    </tr>
    <tr>
      <th>50</th>
      <td>pub_rec_bankruptcies</td>
      <td>Number of public record bankruptcies</td>
      <td>float64</td>
      <td>[0.0, 1.0, 2.0, nan]</td>
    </tr>
    <tr>
      <th>51</th>
      <td>tax_liens</td>
      <td>Number of tax liens</td>
      <td>float64</td>
      <td>[0.0, nan, 1.0]</td>
    </tr>
    <tr>
      <th>52</th>
      <td>hardship_flag</td>
      <td>Flags whether or not the borrower is on a hardship plan</td>
      <td>object</td>
      <td>[N, nan]</td>
    </tr>
    <tr>
      <th>53</th>
      <td>disbursement_method</td>
      <td>The method by which the borrower receives their loan. Possible values are: CASH, DIRECT_PAY</td>
      <td>object</td>
      <td>[Cash, nan]</td>
    </tr>
    <tr>
      <th>54</th>
      <td>debt_settlement_flag</td>
      <td>Flags whether or not the borrower, who has charged-off, is working with a debt-settlement company.</td>
      <td>object</td>
      <td>[N, Y, nan]</td>
    </tr>
  </tbody>
</table>
</div>




```python
def return_month(m_y):
    if m_y is not np.nan:
        return m_y[:3]
```


```python
def return_year(m_y):
    if m_y is not np.nan:
        return int(m_y[-4:])
```


```python
def return_float(percentage):
    if percentage is not np.nan:
        return float(percentage.strip(' %'))/100
```


```python
# Modify the columns mentioned below
# index #4, #29: Values in percentages, the type should be float
# index #13, #23, #41, #43: Are Months and Years
int_rate_float = pd.DataFrame(df_loan.loc[:, 'int_rate'].apply(return_float))
revol_util_float = pd.DataFrame(df_loan.loc[:, 'revol_util'].apply(return_float))
issue_m = pd.DataFrame(df_loan.loc[:, 'issue_d'].apply(return_month)).rename({'issue_d' : 'issue_m'}, axis = 1)
issue_y = pd.DataFrame(df_loan.loc[:, 'issue_d'].apply(return_year)).rename({'issue_d' : 'issue_y'}, axis = 1)
earliest_cr_line_m = pd.DataFrame(df_loan.loc[:, 'earliest_cr_line'].apply(return_month)).rename({'earliest_cr_line' : 'earliest_cr_line_m'}, axis = 1)
earliest_cr_line_y = pd.DataFrame(df_loan.loc[:, 'earliest_cr_line'].apply(return_year)).rename({'earliest_cr_line' : 'earliest_cr_line_y'}, axis = 1)
last_pymnt_m = pd.DataFrame(df_loan.loc[:, 'last_pymnt_d'].apply(return_month)).rename({'last_pymnt_d' : 'last_pymnt_m'}, axis = 1)
last_pymnt_y = pd.DataFrame(df_loan.loc[:, 'last_pymnt_d'].apply(return_year)).rename({'last_pymnt_d' : 'last_pymnt_y'}, axis = 1)
last_credit_pull_m = pd.DataFrame(df_loan.loc[:, 'last_credit_pull_d'].apply(return_month)).rename({'last_credit_pull_d' : 'last_credit_pull_m'}, axis = 1)
last_credit_pull_y = pd.DataFrame(df_loan.loc[:, 'last_credit_pull_d'].apply(return_year)).rename({'last_credit_pull_d' : 'last_credit_pull_y'}, axis = 1)
```


```python
# Remove the 2 columns with wrong type and concat the rest as usable columns
df_loan.pop('int_rate')      
df_loan.pop('revol_util')
df_loan = pd.concat([df_loan, int_rate_float, revol_util_float, issue_m, issue_y, 
          earliest_cr_line_m, earliest_cr_line_y,
          last_pymnt_m, last_pymnt_y,
          last_credit_pull_m, last_credit_pull_y], axis = 1)
```


```python
df_loan
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>...</th>
      <th>int_rate</th>
      <th>revol_util</th>
      <th>issue_m</th>
      <th>issue_y</th>
      <th>earliest_cr_line_m</th>
      <th>earliest_cr_line_y</th>
      <th>last_pymnt_m</th>
      <th>last_pymnt_y</th>
      <th>last_credit_pull_m</th>
      <th>last_credit_pull_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>4975.000000</td>
      <td>36 months</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1065</td>
      <td>0.8370</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jan</td>
      <td>1985.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Dec</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>2500.000000</td>
      <td>60 months</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>Ryder</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1527</td>
      <td>0.0940</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Apr</td>
      <td>1999.0</td>
      <td>Apr</td>
      <td>2013.0</td>
      <td>Oct</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>2400.000000</td>
      <td>36 months</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1596</td>
      <td>0.9850</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Nov</td>
      <td>2001.0</td>
      <td>Jun</td>
      <td>2014.0</td>
      <td>Jun</td>
      <td>2017.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.000000</td>
      <td>36 months</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>AIR RESOURCES BOARD</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1349</td>
      <td>0.2100</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Feb</td>
      <td>1996.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Apr</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.000000</td>
      <td>60 months</td>
      <td>67.79</td>
      <td>B</td>
      <td>B5</td>
      <td>University Medical Group</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1269</td>
      <td>0.5390</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jan</td>
      <td>1996.0</td>
      <td>Jan</td>
      <td>2017.0</td>
      <td>Apr</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>5000.000000</td>
      <td>36 months</td>
      <td>156.46</td>
      <td>A</td>
      <td>A4</td>
      <td>Veolia Transportaton</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0790</td>
      <td>0.2830</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Nov</td>
      <td>2004.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Feb</td>
      <td>2017.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7000.0</td>
      <td>7000.0</td>
      <td>7000.000000</td>
      <td>60 months</td>
      <td>170.08</td>
      <td>C</td>
      <td>C5</td>
      <td>Southern Star Photography</td>
      <td>8 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1596</td>
      <td>0.8560</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jul</td>
      <td>2005.0</td>
      <td>May</td>
      <td>2016.0</td>
      <td>Dec</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.000000</td>
      <td>36 months</td>
      <td>109.43</td>
      <td>E</td>
      <td>E1</td>
      <td>MKC Accounting</td>
      <td>9 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1864</td>
      <td>0.8750</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jan</td>
      <td>2007.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Dec</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5600.0</td>
      <td>5600.0</td>
      <td>5600.000000</td>
      <td>60 months</td>
      <td>152.39</td>
      <td>F</td>
      <td>F2</td>
      <td>NaN</td>
      <td>4 years</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.2128</td>
      <td>0.3260</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Apr</td>
      <td>2004.0</td>
      <td>Apr</td>
      <td>2012.0</td>
      <td>Oct</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5375.0</td>
      <td>5375.0</td>
      <td>5350.000000</td>
      <td>60 months</td>
      <td>121.45</td>
      <td>B</td>
      <td>B5</td>
      <td>Starbucks</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1269</td>
      <td>0.3650</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Sep</td>
      <td>2004.0</td>
      <td>Nov</td>
      <td>2012.0</td>
      <td>Dec</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6500.0</td>
      <td>6500.0</td>
      <td>6500.000000</td>
      <td>60 months</td>
      <td>153.45</td>
      <td>C</td>
      <td>C3</td>
      <td>Southwest Rural metro</td>
      <td>5 years</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.1465</td>
      <td>0.2060</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jan</td>
      <td>1998.0</td>
      <td>Jun</td>
      <td>2013.0</td>
      <td>Dec</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12000.0</td>
      <td>12000.0</td>
      <td>12000.000000</td>
      <td>36 months</td>
      <td>402.54</td>
      <td>B</td>
      <td>B5</td>
      <td>UCLA</td>
      <td>10+ years</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.1269</td>
      <td>0.6710</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Oct</td>
      <td>1989.0</td>
      <td>Sep</td>
      <td>2013.0</td>
      <td>Nov</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>9000.0</td>
      <td>9000.0</td>
      <td>9000.000000</td>
      <td>36 months</td>
      <td>305.38</td>
      <td>C</td>
      <td>C1</td>
      <td>Va. Dept of Conservation/Recreation</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1349</td>
      <td>0.9170</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Apr</td>
      <td>2004.0</td>
      <td>Jul</td>
      <td>2012.0</td>
      <td>Oct</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.000000</td>
      <td>36 months</td>
      <td>96.68</td>
      <td>B</td>
      <td>B1</td>
      <td>Target</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0991</td>
      <td>0.4310</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jul</td>
      <td>2003.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Apr</td>
      <td>2017.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.000000</td>
      <td>36 months</td>
      <td>325.74</td>
      <td>B</td>
      <td>B2</td>
      <td>SFMTA</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1065</td>
      <td>0.5550</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>May</td>
      <td>1991.0</td>
      <td>Oct</td>
      <td>2013.0</td>
      <td>Oct</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.000000</td>
      <td>36 months</td>
      <td>35.31</td>
      <td>D</td>
      <td>D1</td>
      <td>Internal revenue Service</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1629</td>
      <td>0.8150</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Sep</td>
      <td>2007.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Oct</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.000000</td>
      <td>36 months</td>
      <td>347.98</td>
      <td>C</td>
      <td>C4</td>
      <td>Chin's Restaurant</td>
      <td>4 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1527</td>
      <td>0.7020</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Oct</td>
      <td>1998.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Sep</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3600.0</td>
      <td>3600.0</td>
      <td>3600.000000</td>
      <td>36 months</td>
      <td>109.57</td>
      <td>A</td>
      <td>A1</td>
      <td>Duracell</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0603</td>
      <td>0.1600</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Aug</td>
      <td>1993.0</td>
      <td>May</td>
      <td>2013.0</td>
      <td>May</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>6000.0</td>
      <td>6000.0</td>
      <td>6000.000000</td>
      <td>36 months</td>
      <td>198.46</td>
      <td>B</td>
      <td>B3</td>
      <td>Connection Inspection</td>
      <td>1 year</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.1171</td>
      <td>0.3773</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Oct</td>
      <td>2003.0</td>
      <td>Feb</td>
      <td>2015.0</td>
      <td>Jul</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>9200.0</td>
      <td>9200.0</td>
      <td>9200.000000</td>
      <td>36 months</td>
      <td>280.01</td>
      <td>A</td>
      <td>A1</td>
      <td>Network Interpreting Service</td>
      <td>6 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0603</td>
      <td>0.2310</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jan</td>
      <td>2001.0</td>
      <td>Jul</td>
      <td>2012.0</td>
      <td>Feb</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20250.0</td>
      <td>20250.0</td>
      <td>19142.161077</td>
      <td>60 months</td>
      <td>484.63</td>
      <td>C</td>
      <td>C4</td>
      <td>Archdiocese of Galveston Houston</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1527</td>
      <td>0.8560</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Nov</td>
      <td>1997.0</td>
      <td>Aug</td>
      <td>2015.0</td>
      <td>Jun</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21000.0</td>
      <td>21000.0</td>
      <td>21000.000000</td>
      <td>36 months</td>
      <td>701.73</td>
      <td>B</td>
      <td>B4</td>
      <td>Osram Sylvania</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1242</td>
      <td>0.9030</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Feb</td>
      <td>1983.0</td>
      <td>Sep</td>
      <td>2013.0</td>
      <td>Feb</td>
      <td>2017.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.000000</td>
      <td>36 months</td>
      <td>330.76</td>
      <td>B</td>
      <td>B3</td>
      <td>Value Air</td>
      <td>10+ years</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.1171</td>
      <td>0.8240</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jul</td>
      <td>1985.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Jan</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.000000</td>
      <td>36 months</td>
      <td>330.76</td>
      <td>B</td>
      <td>B3</td>
      <td>Wells Fargo Bank</td>
      <td>5 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1171</td>
      <td>0.9180</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Apr</td>
      <td>2003.0</td>
      <td>Oct</td>
      <td>2013.0</td>
      <td>Mar</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>6000.0</td>
      <td>6000.0</td>
      <td>6000.000000</td>
      <td>36 months</td>
      <td>198.46</td>
      <td>B</td>
      <td>B3</td>
      <td>bmg-educational</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1171</td>
      <td>0.2970</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jun</td>
      <td>2001.0</td>
      <td>Oct</td>
      <td>2012.0</td>
      <td>Oct</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>15000.0</td>
      <td>15000.0</td>
      <td>15000.000000</td>
      <td>36 months</td>
      <td>483.38</td>
      <td>B</td>
      <td>B1</td>
      <td>Winfield Pathology Consultants</td>
      <td>2 years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0991</td>
      <td>0.9390</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Feb</td>
      <td>2002.0</td>
      <td>Sep</td>
      <td>2012.0</td>
      <td>Sep</td>
      <td>2012.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>15000.0</td>
      <td>15000.0</td>
      <td>8725.000000</td>
      <td>36 months</td>
      <td>514.64</td>
      <td>C</td>
      <td>C2</td>
      <td>nyc transit</td>
      <td>9 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1427</td>
      <td>0.5760</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Oct</td>
      <td>2003.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jan</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>5000.000000</td>
      <td>60 months</td>
      <td>123.65</td>
      <td>D</td>
      <td>D2</td>
      <td>Frito Lay</td>
      <td>2 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1677</td>
      <td>0.5950</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Oct</td>
      <td>2003.0</td>
      <td>Dec</td>
      <td>2012.0</td>
      <td>Oct</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>4000.0</td>
      <td>4000.0</td>
      <td>4000.000000</td>
      <td>36 months</td>
      <td>132.31</td>
      <td>B</td>
      <td>B3</td>
      <td>Shands Hospital at the University of Fl</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.1171</td>
      <td>0.3770</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Aug</td>
      <td>1984.0</td>
      <td>Apr</td>
      <td>2013.0</td>
      <td>Jan</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>8500.0</td>
      <td>8500.0</td>
      <td>8500.000000</td>
      <td>36 months</td>
      <td>281.15</td>
      <td>B</td>
      <td>B3</td>
      <td>Oakridge homes</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1171</td>
      <td>0.5910</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Nov</td>
      <td>2006.0</td>
      <td>Dec</td>
      <td>2014.0</td>
      <td>Jan</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>42508</th>
      <td>6000.0</td>
      <td>6000.0</td>
      <td>1200.000000</td>
      <td>36 months</td>
      <td>202.51</td>
      <td>D</td>
      <td>D5</td>
      <td>SUNY- ESF</td>
      <td>2 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1312</td>
      <td>0.4930</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>Dec</td>
      <td>2004.0</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Sep</td>
      <td>2012.0</td>
    </tr>
    <tr>
      <th>42509</th>
      <td>5350.0</td>
      <td>5350.0</td>
      <td>625.000000</td>
      <td>36 months</td>
      <td>180.57</td>
      <td>D</td>
      <td>D5</td>
      <td>Clinton Shop Rite of Hunterdon County</td>
      <td>1 year</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.1312</td>
      <td>NaN</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>Dec</td>
      <td>2006.0</td>
      <td>Feb</td>
      <td>2010.0</td>
      <td>Jul</td>
      <td>2013.0</td>
    </tr>
    <tr>
      <th>42510</th>
      <td>1900.0</td>
      <td>1900.0</td>
      <td>900.000000</td>
      <td>36 months</td>
      <td>61.00</td>
      <td>B</td>
      <td>B4</td>
      <td>Scheduall</td>
      <td>1 year</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0964</td>
      <td>NaN</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Feb</td>
      <td>2008.0</td>
      <td>May</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>42511</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>350.000000</td>
      <td>36 months</td>
      <td>345.18</td>
      <td>E</td>
      <td>E5</td>
      <td>GA-PCOM</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1470</td>
      <td>0.8500</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>Sep</td>
      <td>1999.0</td>
      <td>Aug</td>
      <td>2010.0</td>
      <td>Aug</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>42512</th>
      <td>2000.0</td>
      <td>2000.0</td>
      <td>1275.000000</td>
      <td>36 months</td>
      <td>61.87</td>
      <td>A</td>
      <td>A1</td>
      <td>Tzigane Inc</td>
      <td>7 years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0712</td>
      <td>0.0220</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>Mar</td>
      <td>1984.0</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Jun</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>42513</th>
      <td>6000.0</td>
      <td>6000.0</td>
      <td>650.000000</td>
      <td>36 months</td>
      <td>195.28</td>
      <td>C</td>
      <td>C2</td>
      <td>Yale University</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1059</td>
      <td>0.6600</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>Jan</td>
      <td>1996.0</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>Oct</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>42514</th>
      <td>4400.0</td>
      <td>4400.0</td>
      <td>1400.000000</td>
      <td>36 months</td>
      <td>141.25</td>
      <td>B</td>
      <td>B4</td>
      <td>Brick Township board of education</td>
      <td>2 years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0964</td>
      <td>0.6350</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>Jul</td>
      <td>2004.0</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>May</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>42515</th>
      <td>1200.0</td>
      <td>1200.0</td>
      <td>500.000000</td>
      <td>36 months</td>
      <td>38.17</td>
      <td>B</td>
      <td>B2</td>
      <td>Classic Components</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0901</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Jan</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>42516</th>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>375.000000</td>
      <td>36 months</td>
      <td>164.23</td>
      <td>C</td>
      <td>C4</td>
      <td>Compensation Solutions</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1122</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Apr</td>
      <td>2010.0</td>
      <td>Jan</td>
      <td>2017.0</td>
    </tr>
    <tr>
      <th>42517</th>
      <td>1400.0</td>
      <td>1400.0</td>
      <td>475.000000</td>
      <td>36 months</td>
      <td>45.78</td>
      <td>C</td>
      <td>C3</td>
      <td>Stanford University Libraries, LOCKSS Project</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1091</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Sep</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>42518</th>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>625.000000</td>
      <td>36 months</td>
      <td>34.21</td>
      <td>E</td>
      <td>E3</td>
      <td>Macy's</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1407</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>Mar</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>42519</th>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>300.000000</td>
      <td>36 months</td>
      <td>156.11</td>
      <td>A</td>
      <td>A3</td>
      <td>Diamond Management and Technology Consultants</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0775</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Aug</td>
      <td>2009.0</td>
      <td>Aug</td>
      <td>2009.0</td>
    </tr>
    <tr>
      <th>42520</th>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>225.000000</td>
      <td>36 months</td>
      <td>77.69</td>
      <td>A</td>
      <td>A2</td>
      <td>U.S. Bank</td>
      <td>9 years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0743</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jan</td>
      <td>2008.0</td>
      <td>Jun</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>42521</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>250.000000</td>
      <td>36 months</td>
      <td>93.23</td>
      <td>A</td>
      <td>A2</td>
      <td>NC</td>
      <td>1 year</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0743</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jan</td>
      <td>2008.0</td>
      <td>Jun</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>42522</th>
      <td>2600.0</td>
      <td>2600.0</td>
      <td>575.000000</td>
      <td>36 months</td>
      <td>81.94</td>
      <td>A</td>
      <td>A5</td>
      <td>College Pro Painters</td>
      <td>3 years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0838</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Mar</td>
      <td>2010.0</td>
      <td>Aug</td>
      <td>2017.0</td>
    </tr>
    <tr>
      <th>42523</th>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>625.000000</td>
      <td>36 months</td>
      <td>30.94</td>
      <td>A</td>
      <td>A1</td>
      <td>Mana Products</td>
      <td>6 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0712</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>Apr</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>42524</th>
      <td>1275.0</td>
      <td>1275.0</td>
      <td>0.000000</td>
      <td>36 months</td>
      <td>42.65</td>
      <td>D</td>
      <td>D3</td>
      <td>Infinitely law group</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1249</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>May</td>
      <td>2008.0</td>
      <td>Jan</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>42525</th>
      <td>6450.0</td>
      <td>6450.0</td>
      <td>0.000000</td>
      <td>36 months</td>
      <td>211.85</td>
      <td>C</td>
      <td>C4</td>
      <td>Apto Solutions</td>
      <td>2 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1122</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Jun</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>42526</th>
      <td>10500.0</td>
      <td>10500.0</td>
      <td>275.000000</td>
      <td>36 months</td>
      <td>344.87</td>
      <td>C</td>
      <td>C4</td>
      <td>Town of Plainville</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1122</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Aug</td>
      <td>2008.0</td>
      <td>Nov</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>42527</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>125.000000</td>
      <td>36 months</td>
      <td>95.42</td>
      <td>B</td>
      <td>B2</td>
      <td>Tanks Tavern</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0901</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Jun</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>42528</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>0.000000</td>
      <td>36 months</td>
      <td>95.86</td>
      <td>B</td>
      <td>B3</td>
      <td>NaN</td>
      <td>&lt; 1 year</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.0933</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>May</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>42529</th>
      <td>2000.0</td>
      <td>2000.0</td>
      <td>225.000000</td>
      <td>36 months</td>
      <td>64.50</td>
      <td>B</td>
      <td>B5</td>
      <td>NaN</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0996</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Jul</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>42530</th>
      <td>6500.0</td>
      <td>6500.0</td>
      <td>0.000000</td>
      <td>36 months</td>
      <td>208.66</td>
      <td>B</td>
      <td>B4</td>
      <td>Air Force</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0964</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>May</td>
      <td>2008.0</td>
      <td>Jan</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>42531</th>
      <td>3500.0</td>
      <td>3500.0</td>
      <td>225.000000</td>
      <td>36 months</td>
      <td>113.39</td>
      <td>C</td>
      <td>C1</td>
      <td>NaN</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1028</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Mar</td>
      <td>2008.0</td>
      <td>Feb</td>
      <td>2013.0</td>
    </tr>
    <tr>
      <th>42532</th>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>0.000000</td>
      <td>36 months</td>
      <td>32.11</td>
      <td>B</td>
      <td>B4</td>
      <td>Halping hands company inc.</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0964</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>Sep</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>42533</th>
      <td>2525.0</td>
      <td>2525.0</td>
      <td>225.000000</td>
      <td>36 months</td>
      <td>80.69</td>
      <td>B</td>
      <td>B3</td>
      <td>NaN</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0933</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>May</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>42534</th>
      <td>6500.0</td>
      <td>6500.0</td>
      <td>0.000000</td>
      <td>36 months</td>
      <td>204.84</td>
      <td>A</td>
      <td>A5</td>
      <td>NaN</td>
      <td>&lt; 1 year</td>
      <td>NONE</td>
      <td>...</td>
      <td>0.0838</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>Aug</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>42535</th>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>0.000000</td>
      <td>36 months</td>
      <td>156.11</td>
      <td>A</td>
      <td>A3</td>
      <td>Homemaker</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0775</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>Feb</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>42536</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>42537</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>42538 rows × 63 columns</p>
</div>



An example boxplot was made first as an exploration and we can see that the data has a lot of outliners and it made all the boxplot flat. While a few datapoints are staying at the top of the graph, we can fix this by remove those outliners


```python
df_loan.boxplot(column = 'annual_inc', by = 'grade', figsize = (15, 8), fontsize = 15)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fdbbdd6c668>




![png](output_22_1.png)


We can see that the 75% threshold cap at $82500. Also the mean and median are fairly similar so we can safely cut the data at double the median or the mean. The median will be used here. The data will be cut at 120000


```python
df_loan['annual_inc'].describe()
```




    count    4.253100e+04
    mean     6.913656e+04
    std      6.409635e+04
    min      1.896000e+03
    25%      4.000000e+04
    50%      5.900000e+04
    75%      8.250000e+04
    max      6.000000e+06
    Name: annual_inc, dtype: float64




```python
df_loan_cut = df_loan.query('annual_inc <= 120000')
```


```python
df_loan_cut
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>...</th>
      <th>int_rate</th>
      <th>revol_util</th>
      <th>issue_m</th>
      <th>issue_y</th>
      <th>earliest_cr_line_m</th>
      <th>earliest_cr_line_y</th>
      <th>last_pymnt_m</th>
      <th>last_pymnt_y</th>
      <th>last_credit_pull_m</th>
      <th>last_credit_pull_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>4975.000000</td>
      <td>36 months</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1065</td>
      <td>0.8370</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jan</td>
      <td>1985.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Dec</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>2500.000000</td>
      <td>60 months</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>Ryder</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1527</td>
      <td>0.0940</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Apr</td>
      <td>1999.0</td>
      <td>Apr</td>
      <td>2013.0</td>
      <td>Oct</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>2400.000000</td>
      <td>36 months</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1596</td>
      <td>0.9850</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Nov</td>
      <td>2001.0</td>
      <td>Jun</td>
      <td>2014.0</td>
      <td>Jun</td>
      <td>2017.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.000000</td>
      <td>36 months</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>AIR RESOURCES BOARD</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1349</td>
      <td>0.2100</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Feb</td>
      <td>1996.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Apr</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.000000</td>
      <td>60 months</td>
      <td>67.79</td>
      <td>B</td>
      <td>B5</td>
      <td>University Medical Group</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1269</td>
      <td>0.5390</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jan</td>
      <td>1996.0</td>
      <td>Jan</td>
      <td>2017.0</td>
      <td>Apr</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>5000.000000</td>
      <td>36 months</td>
      <td>156.46</td>
      <td>A</td>
      <td>A4</td>
      <td>Veolia Transportaton</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0790</td>
      <td>0.2830</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Nov</td>
      <td>2004.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Feb</td>
      <td>2017.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7000.0</td>
      <td>7000.0</td>
      <td>7000.000000</td>
      <td>60 months</td>
      <td>170.08</td>
      <td>C</td>
      <td>C5</td>
      <td>Southern Star Photography</td>
      <td>8 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1596</td>
      <td>0.8560</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jul</td>
      <td>2005.0</td>
      <td>May</td>
      <td>2016.0</td>
      <td>Dec</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.000000</td>
      <td>36 months</td>
      <td>109.43</td>
      <td>E</td>
      <td>E1</td>
      <td>MKC Accounting</td>
      <td>9 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1864</td>
      <td>0.8750</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jan</td>
      <td>2007.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Dec</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5600.0</td>
      <td>5600.0</td>
      <td>5600.000000</td>
      <td>60 months</td>
      <td>152.39</td>
      <td>F</td>
      <td>F2</td>
      <td>NaN</td>
      <td>4 years</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.2128</td>
      <td>0.3260</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Apr</td>
      <td>2004.0</td>
      <td>Apr</td>
      <td>2012.0</td>
      <td>Oct</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5375.0</td>
      <td>5375.0</td>
      <td>5350.000000</td>
      <td>60 months</td>
      <td>121.45</td>
      <td>B</td>
      <td>B5</td>
      <td>Starbucks</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1269</td>
      <td>0.3650</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Sep</td>
      <td>2004.0</td>
      <td>Nov</td>
      <td>2012.0</td>
      <td>Dec</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6500.0</td>
      <td>6500.0</td>
      <td>6500.000000</td>
      <td>60 months</td>
      <td>153.45</td>
      <td>C</td>
      <td>C3</td>
      <td>Southwest Rural metro</td>
      <td>5 years</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.1465</td>
      <td>0.2060</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jan</td>
      <td>1998.0</td>
      <td>Jun</td>
      <td>2013.0</td>
      <td>Dec</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12000.0</td>
      <td>12000.0</td>
      <td>12000.000000</td>
      <td>36 months</td>
      <td>402.54</td>
      <td>B</td>
      <td>B5</td>
      <td>UCLA</td>
      <td>10+ years</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.1269</td>
      <td>0.6710</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Oct</td>
      <td>1989.0</td>
      <td>Sep</td>
      <td>2013.0</td>
      <td>Nov</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>9000.0</td>
      <td>9000.0</td>
      <td>9000.000000</td>
      <td>36 months</td>
      <td>305.38</td>
      <td>C</td>
      <td>C1</td>
      <td>Va. Dept of Conservation/Recreation</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1349</td>
      <td>0.9170</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Apr</td>
      <td>2004.0</td>
      <td>Jul</td>
      <td>2012.0</td>
      <td>Oct</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.000000</td>
      <td>36 months</td>
      <td>96.68</td>
      <td>B</td>
      <td>B1</td>
      <td>Target</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0991</td>
      <td>0.4310</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jul</td>
      <td>2003.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Apr</td>
      <td>2017.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.000000</td>
      <td>36 months</td>
      <td>325.74</td>
      <td>B</td>
      <td>B2</td>
      <td>SFMTA</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1065</td>
      <td>0.5550</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>May</td>
      <td>1991.0</td>
      <td>Oct</td>
      <td>2013.0</td>
      <td>Oct</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.000000</td>
      <td>36 months</td>
      <td>35.31</td>
      <td>D</td>
      <td>D1</td>
      <td>Internal revenue Service</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1629</td>
      <td>0.8150</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Sep</td>
      <td>2007.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Oct</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.000000</td>
      <td>36 months</td>
      <td>347.98</td>
      <td>C</td>
      <td>C4</td>
      <td>Chin's Restaurant</td>
      <td>4 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1527</td>
      <td>0.7020</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Oct</td>
      <td>1998.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Sep</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3600.0</td>
      <td>3600.0</td>
      <td>3600.000000</td>
      <td>36 months</td>
      <td>109.57</td>
      <td>A</td>
      <td>A1</td>
      <td>Duracell</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0603</td>
      <td>0.1600</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Aug</td>
      <td>1993.0</td>
      <td>May</td>
      <td>2013.0</td>
      <td>May</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>6000.0</td>
      <td>6000.0</td>
      <td>6000.000000</td>
      <td>36 months</td>
      <td>198.46</td>
      <td>B</td>
      <td>B3</td>
      <td>Connection Inspection</td>
      <td>1 year</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.1171</td>
      <td>0.3773</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Oct</td>
      <td>2003.0</td>
      <td>Feb</td>
      <td>2015.0</td>
      <td>Jul</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>9200.0</td>
      <td>9200.0</td>
      <td>9200.000000</td>
      <td>36 months</td>
      <td>280.01</td>
      <td>A</td>
      <td>A1</td>
      <td>Network Interpreting Service</td>
      <td>6 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0603</td>
      <td>0.2310</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jan</td>
      <td>2001.0</td>
      <td>Jul</td>
      <td>2012.0</td>
      <td>Feb</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20250.0</td>
      <td>20250.0</td>
      <td>19142.161077</td>
      <td>60 months</td>
      <td>484.63</td>
      <td>C</td>
      <td>C4</td>
      <td>Archdiocese of Galveston Houston</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1527</td>
      <td>0.8560</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Nov</td>
      <td>1997.0</td>
      <td>Aug</td>
      <td>2015.0</td>
      <td>Jun</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21000.0</td>
      <td>21000.0</td>
      <td>21000.000000</td>
      <td>36 months</td>
      <td>701.73</td>
      <td>B</td>
      <td>B4</td>
      <td>Osram Sylvania</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1242</td>
      <td>0.9030</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Feb</td>
      <td>1983.0</td>
      <td>Sep</td>
      <td>2013.0</td>
      <td>Feb</td>
      <td>2017.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.000000</td>
      <td>36 months</td>
      <td>330.76</td>
      <td>B</td>
      <td>B3</td>
      <td>Value Air</td>
      <td>10+ years</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.1171</td>
      <td>0.8240</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jul</td>
      <td>1985.0</td>
      <td>Jan</td>
      <td>2015.0</td>
      <td>Jan</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.000000</td>
      <td>36 months</td>
      <td>330.76</td>
      <td>B</td>
      <td>B3</td>
      <td>Wells Fargo Bank</td>
      <td>5 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1171</td>
      <td>0.9180</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Apr</td>
      <td>2003.0</td>
      <td>Oct</td>
      <td>2013.0</td>
      <td>Mar</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>6000.0</td>
      <td>6000.0</td>
      <td>6000.000000</td>
      <td>36 months</td>
      <td>198.46</td>
      <td>B</td>
      <td>B3</td>
      <td>bmg-educational</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1171</td>
      <td>0.2970</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Jun</td>
      <td>2001.0</td>
      <td>Oct</td>
      <td>2012.0</td>
      <td>Oct</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>15000.0</td>
      <td>15000.0</td>
      <td>15000.000000</td>
      <td>36 months</td>
      <td>483.38</td>
      <td>B</td>
      <td>B1</td>
      <td>Winfield Pathology Consultants</td>
      <td>2 years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0991</td>
      <td>0.9390</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Feb</td>
      <td>2002.0</td>
      <td>Sep</td>
      <td>2012.0</td>
      <td>Sep</td>
      <td>2012.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>15000.0</td>
      <td>15000.0</td>
      <td>8725.000000</td>
      <td>36 months</td>
      <td>514.64</td>
      <td>C</td>
      <td>C2</td>
      <td>nyc transit</td>
      <td>9 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1427</td>
      <td>0.5760</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Oct</td>
      <td>2003.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jan</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>5000.000000</td>
      <td>60 months</td>
      <td>123.65</td>
      <td>D</td>
      <td>D2</td>
      <td>Frito Lay</td>
      <td>2 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1677</td>
      <td>0.5950</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Oct</td>
      <td>2003.0</td>
      <td>Dec</td>
      <td>2012.0</td>
      <td>Oct</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>4000.0</td>
      <td>4000.0</td>
      <td>4000.000000</td>
      <td>36 months</td>
      <td>132.31</td>
      <td>B</td>
      <td>B3</td>
      <td>Shands Hospital at the University of Fl</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.1171</td>
      <td>0.3770</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Aug</td>
      <td>1984.0</td>
      <td>Apr</td>
      <td>2013.0</td>
      <td>Jan</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>8500.0</td>
      <td>8500.0</td>
      <td>8500.000000</td>
      <td>36 months</td>
      <td>281.15</td>
      <td>B</td>
      <td>B3</td>
      <td>Oakridge homes</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1171</td>
      <td>0.5910</td>
      <td>Dec</td>
      <td>2011.0</td>
      <td>Nov</td>
      <td>2006.0</td>
      <td>Dec</td>
      <td>2014.0</td>
      <td>Jan</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>42502</th>
      <td>2600.0</td>
      <td>2600.0</td>
      <td>950.000000</td>
      <td>36 months</td>
      <td>89.35</td>
      <td>E</td>
      <td>E4</td>
      <td>Department of Veterans Affairs</td>
      <td>10+ years</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.1438</td>
      <td>0.5240</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>Apr</td>
      <td>1992.0</td>
      <td>Jan</td>
      <td>2009.0</td>
      <td>Feb</td>
      <td>2017.0</td>
    </tr>
    <tr>
      <th>42503</th>
      <td>10500.0</td>
      <td>4000.0</td>
      <td>1150.000000</td>
      <td>36 months</td>
      <td>142.43</td>
      <td>G</td>
      <td>G2</td>
      <td>VUTEC, CORP</td>
      <td>4 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1691</td>
      <td>0.2560</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>Dec</td>
      <td>2001.0</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Jul</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>42504</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>1450.000000</td>
      <td>36 months</td>
      <td>325.46</td>
      <td>C</td>
      <td>C2</td>
      <td>NaN</td>
      <td>2 years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.1059</td>
      <td>0.2480</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>Oct</td>
      <td>1998.0</td>
      <td>Apr</td>
      <td>2008.0</td>
      <td>Apr</td>
      <td>2016.0</td>
    </tr>
    <tr>
      <th>42505</th>
      <td>20000.0</td>
      <td>20000.0</td>
      <td>700.000000</td>
      <td>36 months</td>
      <td>693.45</td>
      <td>F</td>
      <td>F1</td>
      <td>hf palm corp</td>
      <td>2 years</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.1501</td>
      <td>0.5930</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>Sep</td>
      <td>1997.0</td>
      <td>Jun</td>
      <td>2008.0</td>
      <td>Jan</td>
      <td>2009.0</td>
    </tr>
    <tr>
      <th>42506</th>
      <td>6725.0</td>
      <td>6725.0</td>
      <td>825.000000</td>
      <td>36 months</td>
      <td>226.98</td>
      <td>D</td>
      <td>D5</td>
      <td>MCHCP</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.1312</td>
      <td>0.8890</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>May</td>
      <td>1991.0</td>
      <td>Apr</td>
      <td>2009.0</td>
      <td>Jan</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>42507</th>
      <td>2000.0</td>
      <td>2000.0</td>
      <td>1025.000000</td>
      <td>36 months</td>
      <td>67.20</td>
      <td>D</td>
      <td>D4</td>
      <td>Signs by Tomorrow</td>
      <td>&lt; 1 year</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.1280</td>
      <td>0.6190</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>Apr</td>
      <td>2004.0</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Jan</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>42508</th>
      <td>6000.0</td>
      <td>6000.0</td>
      <td>1200.000000</td>
      <td>36 months</td>
      <td>202.51</td>
      <td>D</td>
      <td>D5</td>
      <td>SUNY- ESF</td>
      <td>2 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1312</td>
      <td>0.4930</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>Dec</td>
      <td>2004.0</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Sep</td>
      <td>2012.0</td>
    </tr>
    <tr>
      <th>42509</th>
      <td>5350.0</td>
      <td>5350.0</td>
      <td>625.000000</td>
      <td>36 months</td>
      <td>180.57</td>
      <td>D</td>
      <td>D5</td>
      <td>Clinton Shop Rite of Hunterdon County</td>
      <td>1 year</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.1312</td>
      <td>NaN</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>Dec</td>
      <td>2006.0</td>
      <td>Feb</td>
      <td>2010.0</td>
      <td>Jul</td>
      <td>2013.0</td>
    </tr>
    <tr>
      <th>42510</th>
      <td>1900.0</td>
      <td>1900.0</td>
      <td>900.000000</td>
      <td>36 months</td>
      <td>61.00</td>
      <td>B</td>
      <td>B4</td>
      <td>Scheduall</td>
      <td>1 year</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0964</td>
      <td>NaN</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Feb</td>
      <td>2008.0</td>
      <td>May</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>42511</th>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>350.000000</td>
      <td>36 months</td>
      <td>345.18</td>
      <td>E</td>
      <td>E5</td>
      <td>GA-PCOM</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1470</td>
      <td>0.8500</td>
      <td>Jul</td>
      <td>2007.0</td>
      <td>Sep</td>
      <td>1999.0</td>
      <td>Aug</td>
      <td>2010.0</td>
      <td>Aug</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>42513</th>
      <td>6000.0</td>
      <td>6000.0</td>
      <td>650.000000</td>
      <td>36 months</td>
      <td>195.28</td>
      <td>C</td>
      <td>C2</td>
      <td>Yale University</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1059</td>
      <td>0.6600</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>Jan</td>
      <td>1996.0</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>Oct</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>42514</th>
      <td>4400.0</td>
      <td>4400.0</td>
      <td>1400.000000</td>
      <td>36 months</td>
      <td>141.25</td>
      <td>B</td>
      <td>B4</td>
      <td>Brick Township board of education</td>
      <td>2 years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0964</td>
      <td>0.6350</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>Jul</td>
      <td>2004.0</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>May</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>42515</th>
      <td>1200.0</td>
      <td>1200.0</td>
      <td>500.000000</td>
      <td>36 months</td>
      <td>38.17</td>
      <td>B</td>
      <td>B2</td>
      <td>Classic Components</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0901</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Jan</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>42516</th>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>375.000000</td>
      <td>36 months</td>
      <td>164.23</td>
      <td>C</td>
      <td>C4</td>
      <td>Compensation Solutions</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1122</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Apr</td>
      <td>2010.0</td>
      <td>Jan</td>
      <td>2017.0</td>
    </tr>
    <tr>
      <th>42517</th>
      <td>1400.0</td>
      <td>1400.0</td>
      <td>475.000000</td>
      <td>36 months</td>
      <td>45.78</td>
      <td>C</td>
      <td>C3</td>
      <td>Stanford University Libraries, LOCKSS Project</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1091</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Sep</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>42518</th>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>625.000000</td>
      <td>36 months</td>
      <td>34.21</td>
      <td>E</td>
      <td>E3</td>
      <td>Macy's</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1407</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>Mar</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>42520</th>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>225.000000</td>
      <td>36 months</td>
      <td>77.69</td>
      <td>A</td>
      <td>A2</td>
      <td>U.S. Bank</td>
      <td>9 years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0743</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jan</td>
      <td>2008.0</td>
      <td>Jun</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>42521</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>250.000000</td>
      <td>36 months</td>
      <td>93.23</td>
      <td>A</td>
      <td>A2</td>
      <td>NC</td>
      <td>1 year</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0743</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jan</td>
      <td>2008.0</td>
      <td>Jun</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>42522</th>
      <td>2600.0</td>
      <td>2600.0</td>
      <td>575.000000</td>
      <td>36 months</td>
      <td>81.94</td>
      <td>A</td>
      <td>A5</td>
      <td>College Pro Painters</td>
      <td>3 years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0838</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Mar</td>
      <td>2010.0</td>
      <td>Aug</td>
      <td>2017.0</td>
    </tr>
    <tr>
      <th>42523</th>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>625.000000</td>
      <td>36 months</td>
      <td>30.94</td>
      <td>A</td>
      <td>A1</td>
      <td>Mana Products</td>
      <td>6 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0712</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>Apr</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>42524</th>
      <td>1275.0</td>
      <td>1275.0</td>
      <td>0.000000</td>
      <td>36 months</td>
      <td>42.65</td>
      <td>D</td>
      <td>D3</td>
      <td>Infinitely law group</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1249</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>May</td>
      <td>2008.0</td>
      <td>Jan</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>42525</th>
      <td>6450.0</td>
      <td>6450.0</td>
      <td>0.000000</td>
      <td>36 months</td>
      <td>211.85</td>
      <td>C</td>
      <td>C4</td>
      <td>Apto Solutions</td>
      <td>2 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1122</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Jun</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>42526</th>
      <td>10500.0</td>
      <td>10500.0</td>
      <td>275.000000</td>
      <td>36 months</td>
      <td>344.87</td>
      <td>C</td>
      <td>C4</td>
      <td>Town of Plainville</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.1122</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Aug</td>
      <td>2008.0</td>
      <td>Nov</td>
      <td>2018.0</td>
    </tr>
    <tr>
      <th>42527</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>125.000000</td>
      <td>36 months</td>
      <td>95.42</td>
      <td>B</td>
      <td>B2</td>
      <td>Tanks Tavern</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0901</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Jun</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>42528</th>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>0.000000</td>
      <td>36 months</td>
      <td>95.86</td>
      <td>B</td>
      <td>B3</td>
      <td>NaN</td>
      <td>&lt; 1 year</td>
      <td>OWN</td>
      <td>...</td>
      <td>0.0933</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>May</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>42529</th>
      <td>2000.0</td>
      <td>2000.0</td>
      <td>225.000000</td>
      <td>36 months</td>
      <td>64.50</td>
      <td>B</td>
      <td>B5</td>
      <td>NaN</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0996</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jul</td>
      <td>2010.0</td>
      <td>Jul</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>42530</th>
      <td>6500.0</td>
      <td>6500.0</td>
      <td>0.000000</td>
      <td>36 months</td>
      <td>208.66</td>
      <td>B</td>
      <td>B4</td>
      <td>Air Force</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0964</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>May</td>
      <td>2008.0</td>
      <td>Jan</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>42532</th>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>0.000000</td>
      <td>36 months</td>
      <td>32.11</td>
      <td>B</td>
      <td>B4</td>
      <td>Halping hands company inc.</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0964</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>Sep</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>42533</th>
      <td>2525.0</td>
      <td>2525.0</td>
      <td>225.000000</td>
      <td>36 months</td>
      <td>80.69</td>
      <td>B</td>
      <td>B3</td>
      <td>NaN</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>...</td>
      <td>0.0933</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>May</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>42535</th>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>0.000000</td>
      <td>36 months</td>
      <td>156.11</td>
      <td>A</td>
      <td>A3</td>
      <td>Homemaker</td>
      <td>10+ years</td>
      <td>MORTGAGE</td>
      <td>...</td>
      <td>0.0775</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2007.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>Jun</td>
      <td>2010.0</td>
      <td>Feb</td>
      <td>2015.0</td>
    </tr>
  </tbody>
</table>
<p>38969 rows × 63 columns</p>
</div>



Of course now the plots cap at 120000 and that does not tell us much. However, it does not show much about the relation between annual incomes and the grade of the customers either.


```python
df_loan_cut.boxplot(column = 'annual_inc', by = 'grade', figsize = (15, 8), fontsize = 15)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fdbbc33a5f8>




![png](output_28_1.png)


This is bad because the ranking is all over the place. But this plot does show a slight increase in income over the year of employment. We can fix this graph a little


```python
df_loan_cut.boxplot(column = 'annual_inc', by = 'emp_length', figsize = (15, 8), fontsize = 15)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fdbbc33ff28>




![png](output_30_1.png)



```python
def return_emp_year(emp_length):
    if emp_length is not np.nan:
        if emp_length.startswith('<'):
            return 0
        else:
            return int(emp_length[:3].strip(' +y'))
```


```python
emp_length = pd.DataFrame(df_loan_cut.loc[:, 'emp_length'].apply(return_emp_year))
```


```python
df_loan_cut.pop('emp_length')
df_loan_cut = pd.concat([df_loan_cut, emp_length], axis = 1)
```

Now we can see the increases of annual income as the year goes by. The rate of increases is more significant now with the right order. There is also a noticible pause from 4th year to 6th year.


```python
df_loan_cut.boxplot(column = 'annual_inc', by = 'emp_length', figsize = (15, 8), fontsize = 15)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fdbbc2f1940>




![png](output_35_1.png)


Now we're talking. The data is missing out a portion "NONE" category as is has quite a different behavior.<br>
There are quite a lot of high income appliers in this "NONE" group, unfortunately we are missing them here<br>
Meanwhile, those who either own or rent their apartment are at the lower end of the income level. This is even more true with the "RENT" group as the box is rather condensed.<br>
The "MORTGAGE" group is at a middle range.


```python
df_loan_cut.boxplot(column = 'annual_inc', by = 'home_ownership', figsize = (15, 8), fontsize = 15)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fdbbc14e160>




![png](output_37_1.png)


In this plot, those that have not verified their income level are at the lower end, though they are quite similar for those who has their source verified as well.

The best way for the bank to trust you is to have it verified directly, with payment-statement perhaps?


```python
df_loan_cut.boxplot(column = 'annual_inc', by = 'verification_status', figsize = (15, 8), fontsize = 15)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fdbb77c4518>




![png](output_39_1.png)


The "Fully Paid" group has a slightly higher income though not significant. The spread in the data in the other 3 groups does not help much as well. They have a similar median and only a slight differences in first and third quartile


```python
axes = df_loan_cut.boxplot(column = 'annual_inc', by = 'loan_status', figsize = (15, 8), fontsize = 10)
labels = axes.get_xticklabels()
plt.setp(labels, rotation = 45, horizontalalignment = 'right')
plt.show()
```


![png](output_41_0.png)


I want to see if there is anything related between the income and the amount of loan applied. But yeah that did not help much. Of course, the higher amount of loan will not be applied by the lower income group. There is a fine separation in the scatter plot


```python
# I want to try seaborn here
figure, axes = plt.subplots(figsize = (15, 8))
axes = sns.scatterplot(x = 'loan_amnt', y = 'annual_inc', data = df_loan_cut)
plt.show()
```


![png](output_43_0.png)


The plot below makes the annual income distribution looks like a slightly right skewed normal distribution.

This view is achieved by removing the outliners, it will look rather slim with all the outliners


```python
f, a = plt.subplots(figsize = (15, 8))
a = sns.distplot(df_loan_cut['annual_inc'], hist = True, kde = True, 
             bins = int(120000/10000), color = 'blue',
             hist_kws={'edgecolor':'black'})
a.set(xlabel = 'Annual Income', ylabel = 'Frequency', title = 'Annual Income Distribution')
plt.show()
```


![png](output_45_0.png)



```python
f, a = plt.subplots(figsize = (15, 8))
a = sns.distplot(df_loan_cut['annual_inc'], hist = True, kde = True, 
             bins = int(120000/1000), color = 'blue',
             hist_kws={'edgecolor':'black'})
a.set(xlabel = 'Annual Income', ylabel = 'Frequency', title = 'Annual Income Distribution')
plt.show()
```


![png](output_46_0.png)



```python

```
