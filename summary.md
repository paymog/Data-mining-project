Effects of Employment length
============================
- I was initially interested in whether loan applicants would have a normally distributed value for employment length. Turns out the they. The following are graphs of employment length shown as percentages (as opposed to absolute counts) for accepted and rejected loans respectively. ![employment length accepted][emp_length_accepted] ![employment length rejected][emp_length_rejected]
- It's important to note that the data provided on employment length was categorized as follow:
	- < 1 year
	- 1 year
	- 2 years
	- 3 years
	- ...
	- 9 years
	- 10+ years
- It seems pretty apparent that there is some correlation between employment length and acceptance/rejection. There proportion of applicants with less than 1 year of employment is much higher for rejected loans. Similarly accepted loans have a much higher proportion of applicants with a long employment history.

Income Distribution
===================
- Decided to create a histogram of incomes just for fun
- Generated histogram kinda sucked. I looked into the stats a bit more and found that some outliers were mucking things up. The highest reported income was approximately 7 million. These are the graphs generated with the top 1000 incomes removed: ![30 bar income][income_30_bars] ![50 bar income][income_50_bars]
- Wealthy people apply for loans from lending club. Did not expect this at all
- As expected, most people are applying for loans are (what I suspect to be) middle or lower class.
- Distribution is a skewed normal distribution with a long tail for higher incomes

Interest Rate Distribution
==========================
- Not much to say about inteerst rate distribution. It's not normally distributed. If you can make something up about this that would be awesome. Here's the image: ![intrest rate][interest_rate]

Requested Loan Amounts
======================
- Interestingly, and I think this speaks to the pyschology of people, here are the graphs of requested loan amounts with 30, 40 and 50 bars respectively: ![30 bar loan amount][loan_amount_30_bars] ![40 bar loan amount][loan_amount_40_bars] ![50 bar loan amount][loan_amount_50_bars]
- What I find interesting is that there are spikes of requests for multiples of $5000. There's a spike at 5k, 10k, 15k, 20k, 25k, 30k and 35k. There are other spikes too but I find these spikes most interesting.
- It seems people are most comfortable requesting "even" amounts. Personally, I know that I'd be more likely to loan money to someone request 5k vs someone requesting $4634.87
- We see the same thing for rejected loans too ![50 bar loan amounts for rejected loans][loan_amount_50_bars_rejected]

Number of Open Accounts
=======================
- Similar to interest rates there isn't much to say here. High sigma and skewed. There's nothing particularly interesting I can deduce from this. Here's the graph: ![Number of open accounts][open_accounts]

Risk Score Distribution
======================
Like above I don't see anything particularly interesting here. If you can find something worth mentioning, awesome! :) Here's the graph ![Risk scores for rejected loans][risk_score]

Statewide Applications
======================
- Here's the number of applications per state normalized for population for accepted an rejected loans respectively ![Population normalized accepted applications per state][normalized_state_accepted] ![Population normalized rejected applications per state][normalized_state_rejected]
- Here's a table of state name to state number

State Name    			| State Number on graph | State Name    			| State Number on graph
-----------------------	| --------------------- | ----------------------	| ---------------------
Alabama					| 00					| Nebraska					| 26
Alaska					| 01					| Nevada					| 27
Arizona					| 02					| New Hampshire				| 28
Arkansas				| 03					| New Jersey				| 29
California				| 04					| New Mexico				| 30
Colorado				| 05					| New York					| 31
Connecticut				| 06					| North Carolina			| 32
Delaware				| 07					| Ohio						| 33
District of Columbia	| 08					| Oklahoma					| 34
Florida					| 09					| Oregon					| 35
Georgia					| 10					| Pennsylvania				| 36
Hawaii					| 11					| Rhode Island				| 37
Idaho					| 12					| South Carolina			| 38
Illinois				| 13					| South Dakota				| 39
Indiana					| 14					| Tennessee					| 40
Iowa					| 15					| Texas						| 41
Kansas					| 16					| Utah						| 42
Kentucky				| 17					| Vermont					| 43
Louisiana				| 18					| Virginia					| 44
Maryland				| 19					| Washington				| 45
Massachusetts			| 20					| West Virginia				| 46
Michigan				| 21					| Wisconsin					| 47
Minnesota				| 22					| Wyoming					| 48
Mississippi				| 23					| Maine						| 49
Missouri				| 24					| North Dakota				| 50
Montana					| 25					

- The following states have significanly fewer applications per person compared to other states: Florida, Georgia, South Dakota, Texas, Utah, North Dakota. Illinois has significantly more applications per person compared to other states. I can't say why these trends appear. What's interesting to me is that Utah, Georgia and Texas aren't considered (at least to me) progressive states and that may explain why they have fewer applications per person. It might be that the oil boom in the Dakotas expalin why they don't have many applications. I have no possible explanation for Florida or Illinois.

Predicting Loan Status
======================
- Tried to predict the loan status of accepted applications using:
	- Loan Amount
	- Term
	- Interest Rate
	- Grade
	- Employment Length
	- Home ownership status
	- Annual Income
	- Delinquencies in the last 2 years
	- Number of open accounts
- Acheived 70% accuracy with a depth of 12.
- Clustered loan amounts into 10 clusters using kmeans
- Even though we acheived 68% accuracy the classifier isn't great because everything most applications had a status of "Current" which was correctly classified ~63000/65000 times. This classifier did not work out well. Increasing the number of loan amount clusters make the accuracy much worse.
- Here's the table:

True Labels			| Charged Off	| Default	| Issued	| Fully Paid	| Current	| Late (16-30 days)	| Late (31-120 days)	| In Grace Period 
--------------------|---------------|-----------|-----------|---------------|-----------|-------------------|-----------------------|-----------------
Charged Off			| 			  53|		   0| 	       0| 		     191|	    5182| 			       7| 				       7| 			     3
Default				| 			   2|		   0| 	       0| 		       1|	     137| 			       0| 				       0| 			     0
Issued				| 			   0|		   0| 	       0| 		       0|	       1| 			       0| 				       0| 			     0
Fully Paid			| 			 118|		   0| 	       0| 		     452|	   19520| 			       5| 				      21| 			    14
Current				| 			 269|		   1| 	       0| 		    1136|	   63896| 			      17| 				      72| 			    27
Late (16-30 days)	| 			   1|		   0| 	       0| 		       8|	     366| 			       0| 				       2| 			     0
Late (31-120 days)	| 			  19|		   0| 	       0| 		      48|	    1655| 			       2| 				       3| 			     2
In Grace Period		| 			   5|		   0| 	       0| 		      14|	     609| 			       0| 				       1| 			     1

- Most misclassifications happen as a "Current" classification. Most correct classificaitons happen for "Current" also.

Predicting Grade
================
- Firs attempt to predict grade involved using the following predictors:
	- Loan amount (clustered)
	- Term length
	- Employment length
	- home ownership status
	- annual income
	- income verification status
	- DTI
	- Delinquencies in the last two years
	- Number of open accounts
- Achieved 40% accuracy with depth of 8. Results are not good.


True Labels	|	A 	|	B 	|	C 	|	D 	|	E 	|	F 	|	G
------------|-------|-------|-------|-------|-------|-------|-------
A			|3588	|  9471	|   970	|    54	|     2	|     0	|     0
B			|2936	| 22819	|  5124	|   231	|   133	|    18	|     0
C			|1035	| 12703	| 10053	|   439	|   616	|   102	|     0
D			| 474	|  6666	|  5747	|   428	|   601	|   103	|     0
E			|  84	|  1262	|  3440	|   304	|   776	|   179	|     0
F			|  14	|   246	|  1756	|   186	|   554	|   174	|     0
G			|   2	|    16	|   318	|    55	|   124	|    65	|     0

- Here's what the table looks like with interest rate included as a predictor:

True Labels	|	A 	|	B 	|	C 	|	D 	|	E 	|	F 	|	G
------------|-------|-------|-------|-------|-------|-------|-------
A			|14084 	|    1	|     0	|     0	|     0	|     0	|     0
B			|    0 	|31258	|     1	|     0	|     0	|     2	|     0
C			|    0 	|    2	| 24940	|     6	|     0	|     0	|     0
D			|    0 	|    1	|   312	| 13705	|     0	|     1	|     0
E			|    0 	|    2	|     0	|    48	|  5744	|   251	|     0
F			|    0 	|    0	|     0	|     0	|    55	|  2875	|     0
G			|    0 	|    0	|     0	|     0	|     0	|   115	|   465

- This has 99  percent accuracy and does a good gob on all of the grades
- This made me think that there may be some link between grade and interest rate for lending club. A quick google returned [this link][lending_club_link] which shows there there is indeed a link between the two.

Prediction Risk Score
=====================
- Tried to predict risk score with the following variables:
	- Loan Amount (clustered with kmeans)
	- Debt to income ratio
	- employment length
- Acheived accuracy of 28% with depth of 8.
- The risk scores were put into 10 clusters.
- There isn't enough information to predict the risk score, which makes sense. Lending club wouldn't want to give up whatever proprietary algorithm they have.
- Here's the table

True Labels	|	0 	|	1 	|	2 	|	3 	|	4 	|	5 	|	6 	|	7 	|	8 	|	9
------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------
0			|9732	|  2120	|  7354	|    21	| 18431	|   644	|   160	|  4212	|   144	| 20609
1			|1997	| 21324	|   156	|    25	|   392	|   508	|    31	|   782	|    63	|  1253
2			|3835	|  1001	| 49024	|     5	| 25568	|   156	|   201	|  1255	|   874	| 10394
3			|5533	|  2570	|  2317	|    25	|  5205	|   797	|    64	|  3453	|    46	|  8025
4			|6187	|  1296	| 33453	|    12	| 39674	|   274	|   172	|  1988	|   515	| 21388
5			|4435	|  2703	|  1057	|     9	|  2707	|   891	|    53	|  2938	|    46	|  5059
6			| 989	|   565	|  6234	|     7	|  3267	|   115	|   383	|   355	|   279	|  1850
7			|7842	|  2519	|  3872	|    36	|  9511	|   775	|    98	|  4219	|    86	| 13211
8			|2550	|  1078	| 32804	|     6	| 14341	|   136	|   228	|   737	|   958	|  5986
9			|8495	|  1183	| 13720	|     9	| 31295	|   402	|   148	|  3103	|   249	| 25444




[emp_length_accepted]: histograms/emp_length_accept_normed.png "Normalized employment length of accepted loan applicants"
[emp_length_rejected]: histograms/emp_length_reject_normed.png "Normalized employment length of rejected loan applicants"
[income_30_bars]: histograms/income_30_bars.png "Income distribution with 30 bars"
[income_50_bars]: histograms/income_50_bars.png "Income distribution with 50 bars"
[interest_rate]: histograms/interest_rate_20_bars.png "Interest rate for accepted loans"
[loan_amount_30_bars]: histograms/loan_amount_30_bars.png "Loan amoutns with 30 bars"
[loan_amount_40_bars]: histograms/loan_amount_40_bars.png "Loan amoutns with 40 bars"
[loan_amount_50_bars]: histograms/loan_amount_50_bars.png "Loan amoutns with 50 bars"
[loan_amount_50_bars_rejected]: histograms/reject_loan_amount_50_bars.png "Loan amounts with 50 bars for rejected applications"
[open_accounts]: histograms/open_accounts_30_bars.png "Number of open accounts of accpeted applications"
[risk_score]: histograms/risk_score_50_bars.png "Risk scores for rejected loans"
[normalized_state_accepted]: histograms/accept_state_normalized.png "Number of applications per state normalized for population (accepted applications)"
[normalized_state_rejected]: histograms/reject_state_normalized.png "Number of applications per state normalized for population (rejected applications)"
[lending_club_link]: https://www.lendingclub.com/public/how-we-set-interest-rates.action "Lending club interest rates and grades"