# BookSQL: A Large Scale Text-to-SQL Dataset for Accounting Domain

Rahul Kumar† Amar Raja Dibbu† Shrutendra Harsola∗

Vignesh Subrahmaniam∗ Ashutosh Modi†

† Indian Institute of Technology Kanpur (IIT Kanpur) ∗ Intuit {rahulkumar21,amard21}@iitk.ac.in   
{shrutendra\_harsola,vignesh\_subrahmaniam}@intuit.com {ashutoshm}@cse.iitk.ac.in

## Abstract

Several large-scale datasets (e.g., WikiSQL, Spider) for developing natural language interfaces to databases have recently been proposed. These datasets cover a wide breadth of domains but fall short on some essential domains, such as finance and accounting. Given that accounting databases are used worldwide, particularly by non-technical people, there is an imminent need to develop models that could help extract information from accounting databases via natural language queries. In this resource paper, we aim to fill this gap by proposing a new largescale Text-to-SQL dataset for the accounting and financial domain: BookSQL. The dataset consists of 100k natural language queries-SQL pairs, and accounting databases of 1 million records. We experiment with and analyze existing state-of-the-art models (including GPT-4) for the Text-to-SQL task on BookSQL. We find significant performance gaps, thus pointing towards developing more focused models for this domain.

## 1 Introduction

Relational databases are pervasive in all modernday organizations, from financial establishments to educational institutes. Typically, query languages such as SQL are used to extract the required data from relational databases. However, formulating queries in SQL needs mastery of the language itself; consequently, this excludes people (particularly those without technical background, e.g., financial accountants) who do not know SQL from using databases. It is imperative to develop techniques to address the research question, can relational databases be queried using natural language? In this paper, we take a step toward this goal; in particular, we explore if one could develop a natural language interface for accounting databases. In recent years, several large-scale general-purpose datasets (Deng et al., 2022) have been proposed for developing Text-to-SQL systems<sup>1</sup>, such as Spider (Yu et al., 2018) and WikiSQL (Zhong et al., 2017). Such datasets,<sup>2</sup> though cross-domain, are still not suitable for developing systems that could address real-world business use cases, such as accessing accounting databases via natural language interfaces. The primary reason is that these large-scale datasets have a considerable breadth regarding types of domains. However, they either lack certain domains (such as accounting) or have limited data and query types for specific domains (e.g., financial, sales, and marketing). In this paper, we try to address this gap by proposing a large-scale Text-to-SQL dataset (called BookSQL) for the accounting and business domain. We collaborate with financial experts to create a dataset that reflects actual accounting databases used in the industry.

<table><tr><td>Model</td><td>Spider</td><td>BookSQL</td></tr><tr><td>UniSAr</td><td>70%</td><td>3.8%</td></tr><tr><td>SEDE</td><td>63.2%</td><td>0.0%</td></tr><tr><td>RESDSQL</td><td>80.5%</td><td>10.8%</td></tr></table>

Table 1: Performance (Exact Match Accuracy (c.f. §5)) of pre-trained SOTA Text-to-SQL models on Spider and the proposed BookSQL dataset. As can be observed existing models have very poor performance on Book-SQL indicative of poor domain generalization.

To the best of our knowledge, there is no largescale dataset in the accounting domain that contains granular records of accounting books used in businesses. To give an idea about the scale of usage of accounting databases: there are around 33 million small businesses<sup>3</sup> in the US alone. Most of these businesses use accounting software to maintain their books to keep track of their finances, i.e., money-in transactions (e.g., invoice and sales receipt) and money-out transactions (e.g., expense, purchase order, and bill payment). Additionally, for tax purposes, these books need to follow standard accounting principles like double-entry accounting,<sup>4</sup> hierarchical chart of account structure,<sup>5</sup> and accrual accounting.<sup>6</sup> Transactions in the accounting database span across multiple tables. The corresponding SQL queries can involve complex operations such as aggregations, computing distinct counts, and nested queries to extract information from these. For a novice user, this is not an easy task. Moreover, as observed in our initial experiments (Table 1), existing state-of-the-art (SOTA) Text-to-SQL models trained on Spider have very poor performance on domain-specific BookSQL dataset, pointing towards the need for a accounting domain specific dataset which will further lead to the development of SOTA models. In a nutshell, in this resource paper, we make the following contributions:

<table><tr><td>Dataset</td><td>#Size</td><td>#DB</td><td>#D</td><td>#T/DB</td><td>Domain</td><td>ORDER BY</td><td>GROUP BY</td><td>NESTED</td></tr><tr><td>Spider</td><td>10,181</td><td>200</td><td>138</td><td>5.1</td><td>Cross</td><td>1335</td><td>1491</td><td>844</td></tr><tr><td>WikiSQL</td><td>80,654</td><td>26,521</td><td>-</td><td>1</td><td>Cross</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Advising</td><td>3,898</td><td>208</td><td>1</td><td>10</td><td>Single</td><td>15</td><td>9</td><td>22</td></tr><tr><td>BIRD</td><td>12,751</td><td>95</td><td>37</td><td>7.3</td><td>Cross</td><td>2576</td><td>881</td><td>0</td></tr><tr><td>IMDB</td><td>131</td><td>1</td><td>1</td><td>16</td><td>Single</td><td>10</td><td>6</td><td>1</td></tr><tr><td>Yelp</td><td>128</td><td>1</td><td>1</td><td>7</td><td>Single</td><td>18</td><td>21</td><td>0</td></tr><tr><td>BookSQL</td><td>100k</td><td>1</td><td>1</td><td>7</td><td>Single</td><td>17,529</td><td>11,508</td><td>4,456</td></tr></table>

Table 2: Comparison of benchmark datasets with BookSQL. #Size, #DB, #D, and #T/DB represent the numbers of query-SQL pairs, databases, domains, and the averaged number of tables per domain, respectively. The “-” in the #D column indicates an unknown number of domains. Last 3 columns indicate the query types. Yelp dataset is based on Yelp website, IMDB is based on movie domain and Advising dataset is based on the University Course domain

1. We create a new and large-scale Text-to-SQL financial dataset referred to as BookSQL. The dataset consists of a financial-accounts database of 1 million records. The corresponding natural language queries are designed to address various practical intricacies of the accounting domain. BookSQL has 100k Query-

SQL pairs which is about 1.25 times the existing largest Text-2-SQL dataset: WikiSQL. In particular, for designing the queries, we consulted financial experts to understand various practical use cases.

2. We run existing state-of-the-art models (including GPT-4) for the Text-to-SQL task on BookSQL to see the performance and analyze the shortcomings of the models trained on existing large-scale datasets such as Spider, pointing towards developing specialized models for this domain. We release the dataset and model code via GitHub: https://github. com/Exploration-Lab/BookSQL.

## 2 Related Work

Due to its importance in practical applications, developing natural language interfaces to databases has been an active area of research. Due to space constraints, we cannot cover all the research, and we refer the reader to the survey by Deng et al. (2022). We outline some of the main works in this area in this section. Several datasets have been proposed for Text-to-SQL task in recent years. For example, Spider (Yu et al., 2018) dataset has been proposed; it covers 138 different domains. A large-scale dataset, WikiSQL (Zhong et al., 2017), consisting of 24241 Wikipedia tables, has been created. Similarly, Squall (Shi et al., 2020b), KaggleDBQA (Lee et al., 2021), and BIRD-SQL (Li et al., 2023b) datasets have been generated to evaluate the generalization property of models on unseen domains. Domain-specific datasets have also been proposed, such as those based on Yelp and IMDB (Yaghmazadeh et al., 2017), Advising domain (Finegan-Dollak et al., 2018), MIMICSQL (Wang et al., 2020), SEDE (Hazoom et al., 2021a),

<table><tr><td>BookSQL</td><td>Stats</td></tr><tr><td>Size of the database</td><td>1 million</td></tr><tr><td>Total Businesses</td><td>27</td></tr><tr><td>Size of Question-SQL Pair</td><td>100k</td></tr><tr><td>Number of Easy SQL</td><td>10,000</td></tr><tr><td>Number of Medium SQL</td><td>45,000</td></tr><tr><td>Number of Hard SQL</td><td>45,000</td></tr></table>

Table 3: Statistics of BookSQL.

Restaurants domain (Tang and Mooney, 2001), and Academic domain (Li and Jagadish, 2014). The purpose of these datasets is to evaluate the performance of models with a high degree of precision while disregarding the generalization characteristic of the models.

Comparison. We compare BookSQLwith other popular datasets in Table 2. As can be observed, BookSQL has a much large number of Query-SQL pairs, has a more diverse number of queries in terms of the SQL clauses (e.g., ORDER BY), and involves more complex (and nested) queries. Benchmark dataset such as Spider have a very wide coverage over various domains (138) but very few queries per domain (e.g., average number of queries per domain is 74 in the case of Spider), limiting its performance in a specific domain (see also Table 1). Moreover, BookSQL can be merged with the existing Spider dataset to increase its coverage in the business domain.

Models. Various models have been proposed for the Text-to-SQL task (Deng et al., 2022). Some state-of-the-art models include the non-invasive UniSAr model (Dou et al., 2022) based on Seq2Seq architecture. The model has shown high accuracy on the multi-domain, multi-table Spider dataset. RESDSQL (Li et al., 2023a) decouples the schema linking and the skeleton parsing for Text-to-SQL generation. Schema linking identifies the table and columns required for a given question. Skeleton parsing first generates the SQL skeleton and then the final SQL. It achieves SOTA performance on the Spider benchmark.

## 3 BookSQL Dataset

Given the importance and wide prevalence of business databases across the world, the proposed dataset, BookSQL focuses on the finance and accounting domain. Accounting databases are used across a wide spectrum of industries like construction, healthcare, retail, educational services, insurance, restaurant, real estate, etc. Business in these industries arranges their financial transactions into their own different set of categories (called a chart of accounts<sup>7</sup> in accounting terminology). For example, a restaurant business could have categories like advertising, license fees, etc., a real estate brokerage business could have categories like commissions, office supplies, etc. Keeping generalization in mind BookSQL dataset includes a variety of businesses from different industries. Hence, a Text-to-SQL system developed on BookSQL will be robust at handling various types of accounting databases. The total size of the dataset is 1 million. The dataset is prepared under financial experts’ supervision, and the dataset’s statistics are provided in Table 3. The dataset consists of 27 businesses, and each business has around 35k - 40k transactions. The distributions of all businesses and their products are shown in Appendix Figure 3 and Figure 4.

## 3.1 BookSQL Tables

Figure 1 shows the detailed database schema. The schema is reflective of real-life databases used in the finance and accounting domain. There are seven tables in the BookSQL, namely, Master Transactions, Customer, Employees, Product Service, Vendor, Chart of Account, and Payment Method tables. We arrived at the list of seven tables after examining (and corresponding discussions with finance experts) the databases of several businesses. Given the nature of accounting domain, majority of databases used by businesses across the globe are restricted mainly to these seven tables only. The main table is the “Master Transaction” table (e.g., Appendix Table 8), which records money-in transactions (invoice, sales receipt, etc.) and money-out transactions (expense, purchase order, bill payment, etc.) This table also records additional corresponding transaction details, like the customer, vendor, product/service, credit account, debit account, and amount. The “Chart of accounts” table (e.g., Appendix Table 9) contains information on all account names and types. The “Customer” table (e.g., Appendix Table 10) contains all the customer’s details, i.e., name, billing, and shipping address. The “Vendors” table (e.g., Appendix Table 11) contains all the vendor details of all the businesses, i.e., vendor names and billing addresses. The “Employees” table (e.g., Appendix

![](images/502f1018935128c9e430a802fb5b6eb7f227328163252ee175ca738d779a6386.jpg)  
Figure 1: BookSQL Database schema

Table 12) contains information about all the business employees. The “Product service” table (e.g., Appendix Table 13) contains the details of all the products and services. The “Payment method” table (e.g., Appendix Table 14) contains different payment methods the business uses.

## 3.2 Financial Constraints

For creating the dataset, we took existing accounting databases based on the schema described above and anonymized the names and entries in the tables, i.e., actual names, businesses, and numbers were replaced with fictional ones while adhering to the financial constraints (described next). This is done to maintain the privacy of individuals and businesses. The resulting database is a true reflection of a realworld accounting setting. Accounting databases follow certain accounting rules and financial constraints, which were followed when anonymizing the database. In particular, standard double-entry accounting was followed, which means every entry to an account needs a corresponding and opposite entry to a different account, i.e., debit and credit. So, the sum of debit should always be equal to the sum of credit for every transaction. All seven tables were partitioned by business\_id. For a given transaction\_id, the sum of the credits column should equal the sum of the debits column, and both should equal the amount column in the Master Transactions table. Credit (in the Master Transaction table) should be equal to the product of Quantity and Rate. The chart of accounts was anonymized using the industry-wise list published by a popular CPA.<sup>8</sup> Business-specific custom fields were anonymized using the examples provided in the help articles of various accounting software. The created database was cross-checked with financial experts to make sure that the created database looked like a realworld accounts database.

## 3.3 Dataset Creation and Annotation

BookSQL dataset consists of 100k questions in natural language and their corresponding SQL on multiple tables, covering 27 different businesses. We involved financial experts in the query creation process. We collaborated with two financial experts who have previously been involved in the creation of accounting software. Moreover, these experts have the knowledge and experience in dealing with customer interactions involving account books. The financial experts helped us on a pro bono basis since the creation of Text-to-SQL system for the accounting domain would help them and their customers.

![](images/7a387d398d8602cdbc6cd041d16f3b345a7bc0d647dc5b0cbb62a7c6dab0e937.jpg)  
Figure 2: An example showing the pipeline for creating BookSQL dataset. Note, here we can replace aggregation\_entity by max, min, total, and average, and customer\_name can be replaced with any possible name to get the Question-SQL pair. Similarly, date/period can be replaced with last quarter, this quarter, last month.

The question-SQL pair formulation process is as follows. With the help of financial experts, we first created a list of typical questions (based on the account book) that customers (or business people) usually ask or questions about the information that customers are interested in knowing. We tried to keep the questions (queries) as natural as possible to capture real-world scenarios. We relied on the experience of financial experts to keep the list as exhaustive as possible. We also created the corresponding SQL query for each of the natural language queries in the list. The queries in the list were then used to create more queries via the process of templatization. Figure 2 explains the process with help of an example.

In order to be as exhaustive as possible, with the help of experts, we arrived at a list of 183 unique natural language questions that customers typically ask when interacting with accounting databases. These natural language questions were used to create query templates, and this was further used to generate diverse range of Question-SQL pairs in BookSQL. Additionally, we performed a second round of verification of the BookSQL corpus and query templates with financial experts to verify the consistency, veracity, and ensure that the dataset reflected the real-world scenario. Note that existing general Text-to-SQL datasets (e.g., Spider and WikiSQL) consist of databases from multiple domains and BookSQL is focused on the financial domain, hence the number of templates may appear to be less. However, the number of templates is still large when compared across a single domain, for example, to give a rough estimate, Spider dataset uses 5693 templates and spans 138 domains, so a rough estimate of number of templates per domain is about 41 ( 5693/138). Note that Spider doesn’t provide details about templates for each domain, hence a rough estimate. Moreover, questions in existing Text-to-SQL datasets (like Spider) are created by students (Yu et al., 2018), whereas questions in BookSQL are created by financial experts who use accounting systems on a regular basis and are well-versed. Although our dataset is small (in terms of total number of templates), it is of high quality and more complex; hence it helps in learning models that would generalize well. Moreover, while experimenting with models, the queries in the test set are based on templates that are not used during training (see section 5).

To the best of our knowledge, BookSQLis the first Text-to-SQLdataset to have multi-step questions, which requires nested SQL queries to get the answer. For example - "What products are selling less than last month/week?" It would first require computing monthly/weekly product level sales and then comparing each product’s current and last month’s/week’s sales. BookSQLdatabase schema also contains complex column types. Additionally, BookSQLis the first Text-to-SQLdataset to have extensive time-based filters like last month, this quarter to date, last financial year, between July to August, this week, yesterday, etc.

## 3.4 Complexity of SQL in BookSQL

SQL queries in BookSQL are diverse and cover various levels of complexity, i.e., it covers the following operations: SELECT with multiple columns and aggregations, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, JOIN, INTERSECT, UNION, NOT IN, OR, AND, EXISTS, CONTAINS as well as nested queries. Table 2 shows the comparisons of all Text-to-SQL datasets. In terms of complexity, BookSQL consists of complex SQL queries containing 17,529 ORDER BY, 11,508 GROUP BY, and 4,456 NESTED queries. We further divided all Query-SQL pairs into three categories: Easy, Medium, and Hard, based on the complexity of SQL. Table 4 shows examples for each category. Table 3 shows the main statistics of the BookSQL. BookSQL consists of 7,193 Hard SQL queries, making it a more complex, large, and challenging dataset. We used the following criteria to decide on the complexity of a query.

<table><tr><td>Complexity Question</td><td></td><td>SQL</td></tr><tr><td>Easy</td><td>owned by John?</td><td>What is the balance SELECT balance from Customers where customer_name = &#x27;John&#x27;</td></tr><tr><td>Medium</td><td></td><td>What is the maxi- SELECT MAX(credit) FROM master_txn_table where ac- mum sales for John count_type in (&#x27;Income&#x27;, &#x27;Other Income&#x27;) AND customer = &#x27;John&#x27; in the last month? AND month(transaction_date) = month(current_timestamp) - 1</td></tr><tr><td>Hard</td><td>month?</td><td>What products are SELECT A.product_service, revenue_this_month, rev- selling less than last enue_last_month FROM (SELECT product_service, SUM(credit) as revenue_this_month FROM master_txn_table WHERE account_type in (&#x27;Income&#x27;, &#x27;Other Income&#x27;) AND month(transaction_date) = month(current_timestamp) GROUP BY 1) AS A INNER JOIN (SELECT product_service, SUM(credit) as revenue_last_month FROM master_txn_table WHERE account_type in (&#x27;Income&#x27;, &#x27;Other Income&#x27;) AND month(transaction_date) = month(current_timestamp) - 1 GROUP BY 1) AS B ON A.product_service = B.product_service WHERE</td></tr></table>

Table 4: Examples of Question-SQL pairs from BookSQLbased on complexity of the query.

• EASY: simple queries with single WHERE condition

• MEDIUM: multiple conditions in WHERE clause and multiple columns in SELECT clause

• HARD: Join, Group by, Inner queries, Union, Except as these are hard to predict from Natural Language question.

## 4 Baseline Models

We benchmark existing state-of-the-art (SOTA) Text-to-SQL models on BookSQL dataset. SEDE: We fine-tuned the SEDE model (Hazoom et al., 2021b) on the BookSQL dataset. SEDE is a T5-based sequence-to-sequence model (Raffel et al., 2020). It takes unordered schema items (tables and column names) along with questions as input and generates the corresponding SQL query as output.

UniSAr: We fine-tuned the UniSAr model (Dou et al., 2022) on the BookSQL train dataset, with T5-large as the base language model. UniSAr converts any seq-to-seq language model into a text2sql model by three non-invasive extensions: (1) Structure Mark to encode database schema in the model input, (2) Constrained Decoding to generate wellstructured SQL. For the BookSQL dataset, we removed the constrained decoding module of UniSAr, since it did not support the SQL queries with complex grammar present in the BookSQL dataset. (3) SQL Completion for completing potential missing JOIN relationships.

RESDSQL: RESDSQL (Li et al., 2023a) decouples the schema linking and the skeleton aware decoding for SQL generation. A cross-encoder is trained to rank the tables and columns required for a given query for schema linking. For SQL generation, a seq-to-seq model with skeleton-aware decoding is used, which first generates an SQL skeleton, and then the model predicts the actual SQL query from it. The masked self-attention method in the decoder allows the first created skeleton to direct the future SQL parsing implicitly.

DIN-SQL + GPT4: We use prompt chaining technique as proposed in Pourreza and Rafiei (2023). It decomposes Text-to-SQL task into multiple subtasks and then solves each sub-task one by one by prompting GPT4 (Achiam et al., 2023) with sub-task-specific prompts. It uses the following sub-tasks:

<table><tr><td rowspan="2">Model</td><td colspan="2">Spider</td><td colspan="5">BookSQL</td></tr><tr><td>EMA</td><td>EA</td><td>EMA</td><td>PCM-F1</td><td>EA</td><td>BLEU-4</td><td>ROUGE-L</td></tr><tr><td>SEDE</td><td>63.2%</td><td>1</td><td>43.4%</td><td>0.82</td><td>44.3%</td><td>0.69</td><td>0.83</td></tr><tr><td>UniSAr</td><td>70%</td><td></td><td>43.0%</td><td>0.78</td><td>47.6%</td><td>0.72</td><td>0.80</td></tr><tr><td>RESDSQL</td><td>80.5%</td><td>84.1%</td><td>51.5%</td><td>0.81</td><td>54.4%</td><td>0.74</td><td>0.81</td></tr><tr><td>DIN-SQL+GPT4</td><td>60%</td><td>85.3%</td><td>9.3%</td><td>0.63</td><td>7.6%</td><td>0.43</td><td>0.68</td></tr><tr><td>DFew+GPT4</td><td>一</td><td></td><td>47.5%</td><td>0.89</td><td>67.2%</td><td>0.86</td><td>0.90</td></tr></table>

Table 5: Results on Spider and BookSQL datasets. EMA refers to Exact Match Accuracy, EA refers to Execution Accuracy, and PCM-F1 refers to Partial Component Match F1. DFew+GPT4 refers to Dynamic few-shot prompt+GPT4

1. Schema Linking: This module identifies references to database tables and columns required to answer the natural language question.

2. Classification and Decomposition: This module classifies each question into easy, nonnested complex, and nested complex. This signifies the type of SQL query required for the given question.

3. SQL Generation: This module generates the SQL using the output of previous modules.

4. Self Correction module: This module is responsible for correcting any minor mistakes in the SQL generated by the previous module.

Sample prompts for each of these sub-tasks are provided in the Appendix §C.

Dynamic few-shot prompt + GPT4 (DFew+GPT4): We follow a dynamic fewshot prompting technique similar to Sun et al. (2023). Firstly, a vector database is created by an embedding train set questions using SentenceTransfomers all-MiniLM-L6-v2 model.<sup>9</sup> This model is trained on the 1 billion sentence pairs dataset<sup>10</sup> and is best suited for generating sentence embeddings. This created embedding database is called trainDB. Then, at inference time, embedding for the test question is created using the same SentenceTransfomers model, and this embedding is used to do ANN (Approximate Nearest Neighbor) search in trainDB to get ten examples from the train set. These ten examples and database schema is used to create the few-shot SQL generation prompt for GPT4. Pseudo-code and sample prompts are provided in Appendix §B. We use ChromaDB<sup>11</sup> as the underlying vector database and for ANN search.

## 5 Experiments, Results and Analysis

## 5.1 Evaluation Metrics

We use the standard evaluation metrics (details in Appendix D) of Exact Match Accuracy (EMA) (Yu et al., 2018), Execution Accuracy (EA) (Yu et al., 2018), Partial Component Match F1 (PCM-F1) (Hazoom et al., 2021a), BLEU-4 (Papineni et al., 2002), and ROUGE-L (Lin, 2004).

## 5.2 Experimental Setup

We divide the dataset into 70% train, 10% validation, and 20% test sets based on query templates. The test set contains 14.37% easy, 78.43% medium, and 7.2% hard SQL queries. In order to check the generalization performance, queries in the test set are based on templates that are not used during training. Given limitations on the number of calls to OpenAI GPT4 API, we used a random 10% of BookSQL test set for GPT4-based approaches. We provide details about training and hyper-parameters in Appendix E.

## 5.3 Results

Table 5 and Table 6 shows the performance of baseline models. Table 5 shows the performance of

<table><tr><td>Query</td><td>SEDE</td><td></td><td>UniSAr RESDSQL GPT4</td><td></td></tr><tr><td>E</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>M</td><td>43.08</td><td>46.49</td><td>62.12</td><td>71.35</td></tr><tr><td>H</td><td>15.00</td><td>12.34</td><td>15.00</td><td>22.08</td></tr></table>

Table 6: Execution Accuracy (in %) of various models on SQL queries of varying complexity. E refers to Easy query, M refers to Medium query and H refers to query with Hard complexity.

SOTA Text-to-SQL models fine-tuned on the Book-SQL dataset. RESDSQL performs best as can be observed with regard to exact match accuracy and execution accuracy scores. SEDE and UniSAr have poor exact match and execution accuracy scores. Though BookSQL and Spider are not directly comparable, we also include results of the models on the Spider dataset to provide a reference for comparison purposes. As can be observed, the models that perform well on Spider do not have a good performance on BookSQL, indicating the complexity of the dataset. Table 5 shows the in-context learning performance of GPT4 on the BookSQL test set. DIN-SQL+GPT4 could only get 9.3% exact match accuracy, while Dynamic few-shot prompt+GPT4 comes close to the best-fine-tuned model, with exact match accuracy of 47.5% and execution accuracy of 67.2%. Table 6 shows the performance on easy, medium and hard queries. All models have a perfect performance (100% execution accuracy) on easy queries but struggle with medium and hard queries.

## 5.4 Error Analysis

We observed that SOTA models fail on queries with date filters, nested queries, distinct aggregations, and domain-specific filters. Table 7 shows the outputs of models on some examples from the test set. DIN-SQL + GPT4 performs very poorly with execution accuracy of 7.6%. Perhaps, the reason for the bad performance is that it uses the same static chain-of-thought prompt, irrespective of the test question. BookSQL questions are very diverse and require domain knowledge. It is impossible to capture this diversity and domain knowledge in only a few examples in the prompt. Due to this, DIN-SQL fails whenever the test question is completely different from the examples provided in the prompt. Dynamic few shot prompt + GPT4 model addresses the limitations of DIN-SQL by dynamically selecting a few shot examples for the prompt based on the test question. It significantly improves execution accuracy to 67.2%. Possibly, the reasons for poor performance are: 1) Getting confused between different columns (like WHERE clause on product\_service vs. account column - see table 7), 2) Mixing up credit, debit, and amount columns and using incorrect columns in aggregations, 3) Not able to generate Nested SQL, even when required to answer the test question correctly, 4) failing when domain-specific information is required to generate SQL correctly. For example, transaction\_type filters of the invoice, sales receipt, and purchase order, or account\_type filters of expense, income, account receivable, and account payable are incorrectly applied.

SEDE fails to generate correct SQL, possibly due to a lack of question and schema linking in the input to the T5 model. Due to this, it mixes up different columns like customer, vendor, product\_service, and account. UniSAr performs poorly, possibly due to complex queries introduced in Book-SQL like date filters, nested queries, distinct aggregations, etc. UniSAr introduces constrained grammar-based decoding, which works well for simple queries but fails with such complex queries. RESDSQL is the best-performing model. Poor performance is possibly due to: 1) Failure at complex time-based questions like "What is average revenue for customer X in last 6 years" (see table 7); 2) mixing up of credit and debit columns; 3) failure when distinct aggregations are required like COUNT(DISTINCT transaction\_id); 4) failure in case of many nested queries.

## 6 Future Directions

Results show the poor performance of the SOTA models on BookSQL. We outline some of the possible directions for the future to improve performance.

Multi-task learning: One could employ a multitask learning setup, i.e., in addition to optimizing for SQL generation objective, adding other multitask objectives could help improve the performance on hard SQL queries. These objectives could include (1) nested vs. non-nested SQL classification, (2) distinct keyword classification, and (3) date format classification.

Pre-training: For large databases, it is difficult for any model to relate the question tokens with column names when the question might refer to some table cell value. Before the Text-to-SQL task, one could do pre-training to better understand the question and table relationships. This can be done using mask modeling by defining tasks such as column recovery and column predictions where few tokens could be masked, and the model tries to recover and predict the masked tokens; a similar approach is proposed by Shi et al. (2020a) via the GAP model.

<table><tr><td>Question:</td><td>What was the average invoice value for Biogenic municipal waste-fueled power generation?</td></tr><tr><td>Gold SQL:</td><td>SELECT avg(credit) FROM master_txn_table WHERE transaction_type = &#x27;invoice&#x27; AND</td></tr><tr><td></td><td>instr(account, &#x27;Biogenic municipal waste-fueled power generation&#x27;) Few-shot GPT4: SELECT avg(amount) FROM master_txn_table WHERE transaction_type = &#x27;invoice&#x27; AND</td></tr><tr><td>SEDE:</td><td>productservice = &#x27;Biogenic municipal waste-fueled power generation&#x27;X SELECT avg(credit) FROM master_txn_table WHERE transaction_type = &#x27;invoice&#x27; AND</td></tr><tr><td>UniSAr:</td><td>instr(account, &#x27;biogenic municipal waste-fueled power generation&#x27;) √ SELECT avg(credit) FROM master_txn_table WHERE transaction_type = &#x27;invoice&#x27;AND</td></tr><tr><td>RESDSQL:</td><td>product_service = &#x27;Biogenic municipal waste-fueled power generation&#x27; X SELECT avg(credit) FROM master_txn_table WHERE transaction_type = &#x27;invoice&#x27;AND</td></tr><tr><td>2 Question:</td><td>instr(account, &#x27;biogenic municipal waste-fueled power generation&#x27;) √ What was the last invoice value for Drain cleaning in This week to date?</td></tr><tr><td>Gold SQL:</td><td>SELECT max(credit) FROM master_txn_table WHERE transaction_type = &#x27;invoice&#x27; AND instr(account,&#x27;Drain cleaning&#x27;) AND transaction_date BETWEEN date(current_date, &#x27;weekday</td></tr><tr><td></td><td>0&#x27;, &#x27;-7 days&#x27;) AND date( current_date) Few-shot GPT4: SELECT credit FROM master_txn_table WHERE transaction_type = &#x27;invoice&#x27; AND prod- uct_service = &#x27;Drain cleaning&#x27; AND transaction_date BETWEEN date(current_date, &#x27;weekday</td></tr><tr><td>SEDE:</td><td>0&#x27;, &#x27;-7 days&#x27;) AND date(current_date) ORDER BY transaction_date DESC LIMIT 1 X SELECT max(credit) FROM master_txn_table WHERE transaction_type = &#x27;invoice&#x27; AND customers = &#x27;drain cleaning&#x27; AND transaction_date BETWEEN date(current_date, &#x27;weekday</td></tr><tr><td>UniSAr:</td><td>0&#x27;, &#x27;-7 days&#x27;) AND date(current_date) X SELECT max(credit) FROM master_txn_table WHERE transaction_date BETWEEN date (</td></tr><tr><td>RESDSQL:</td><td>current_date , &#x27;weekday 0&#x27;, &#x27;-7 days&#x27;) AND date (current_date) X SELECT max (credit) FROM master_txn_table WHERE transaction_type = &#x27;invoice&#x27; AND instr (account , &#x27;Drain cleaning’) AND transaction_date BETWEEN date (current_date ,</td></tr><tr><td>3 Question:</td><td>&#x27;weekday 0&#x27;, &#x27;-7 days&#x27;) AND date (current_date ) √ What is my average revenue for Customer Nathan Hernandez in the last 6 years?</td></tr><tr><td>Gold SQL:</td><td>SELECT sum(credit)/6 FROM master_txn_table WHERE customers = &#x27;Nathan Hernandez&#x27; AND strftime(&#x27;%Y&#x27;, transaction_date) &gt;= strftime(&#x27;%Y&#x27;, current_date)-6-1</td></tr><tr><td></td><td>Few-shot GPT4: SELECT sum(credit)/6 FROM master_txn_table WHERE customers = &#x27;Nathan Hernandez&#x27; AND strftime(&#x27;%Y&#x27;, transaction_date) &gt;= strftime(&#x27;%Y&#x27;, current_date)-6-1 √</td></tr><tr><td>SEDE:</td><td>SELECT avg(credit) FROM master_txn_table AS t1 JOIN chart_of_accounts AS t2 ON t1.account = t2.account_name WHERE account_type IN (&#x27;income&#x27;,&#x27;other income&#x27;) AND cus-</td></tr><tr><td>UniSAr:</td><td>tomers = &#x27;nathan hernandez&#x27; AND transaction_date BETWEEN date(current_date,&#x27;start of year&#x27;,&#x27;-6 year&#x27;) AND date(current_date,&#x27;start of year&#x27;,-1 day&#x27;) X SELECT avg(credit) FROM master_txn_table as T1 JOIN chart_of_accounts as T2 on T1.account = T2.account_name WHERE account_type in (&#x27;Income&#x27; , &#x27;Other Income&#x27;) AND</td></tr><tr><td>RESDSQL:</td><td>customers = &#x27;Nathan Hernandez&#x27; AND transaction_date BETWEEN date (current_date ,&#x27;start of year&#x27; , &#x27;-6 year&#x27;) AND date (current_date ,&#x27;start of year&#x27; , &#x27;-1 day&#x27;) X SELECT sum (credit) / 6 FROM master_txn_table WHERE customers = &#x27;Nathan Hernandez&#x27; AND strftime  $\big ( \mathrm { \because } \mathrm { \% } \mathrm { Y } ^ { \mathrm { \prime } }$  , transaction_date ) &gt;= strftime  $\mathbf { \Xi } ( \mathbf { \Lambda } ^ { \bullet } \% \mathbf { Y } ^ { \bullet }$  , current_date ) - 1 GROUP BY</td></tr></table>

Table 7: Error analysis for different models on BookSQLtest set

Multi-step few-shot prompting: One could also generate SQL in multiple steps using dynamic fewshot prompting instead of generating in a single step.

Value Encoding: In-context learning models (GPT4) mixes up different columns due to a lack of knowledge about table contents. Adding related table rows in the prompt could alleviate this issue.

## 7 Conclusion

In this paper, we propose BookSQL, a Text-to-SQL dataset that will have broad applications in the finance and accounting domain. The experimental outcomes of several Text-to-SQL models indicate considerable room for improvement. In the future, we aim to build a more robust model that can handle hard queries and improve performance.

## Limitations

Since this is a resource paper, we release a large dataset and consequently focus less on modeling the Text-to-SQL system. We tested existing Text-to-SQL systems to see how well these fare on the new dataset. The results are indicative of considerable scope for improvement. In the future, we will focus on developing new models with better performance on BookSQL. Moreover, we hope that once the dataset is released, it will foster more research in this domain, resulting in more interesting models.

## Ethics Statement

Considering the privacy aspect, we create anonymized entries in the dataset. Moreover, the dataset was verified by financial experts to make sure that the entries adhere to accounting principles and are reflective of real-life scenarios. We will be releasing the dataset publicly for research uses. To the best of our knowledge, we are not aware of any other possible ethical consequences of the proposed dataset.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Naihao Deng, Yulong Chen, and Yue Zhang. 2022. Recent advances in text-to-SQL: A survey of what we have and what we expect. In COLING, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.

Longxu Dou, Yan Gao, Mingyang Pan, Dingzirui Wang, Wanxiang Che, Dechen Zhan, and Jian-Guang Lou. 2022. Unisar: A unified structure-aware autoregressive language model for text-to-sql.

Catherine Finegan-Dollak, Jonathan K. Kummerfeld, Li Zhang, Karthik Ramanathan, Sesh Sadasivam, Rui Zhang, and Dragomir Radev. 2018. Improving textto-SQL evaluation methodology. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 351–360, Melbourne, Australia. Association for Computational Linguistics.

Moshe Hazoom, Vibhor Malik, and Ben Bogin. 2021a. Text-to-SQL in the wild: A naturally-occurring dataset based on stack exchange data. In Proceedings of the 1st Workshop on Natural Language Processing for Programming (NLP4Prog 2021), pages 77–87, Online. Association for Computational Linguistics.

Moshe Hazoom, Vibhor Malik, and Ben Bogin. 2021b. Text-to-sql in the wild: a naturally-occurring dataset based on stack exchange data. arXiv preprint arXiv:2106.05006.

Chia-Hsuan Lee, Oleksandr Polozov, and Matthew Richardson. 2021. KaggleDBQA: Realistic evaluation of text-to-SQL parsers. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2261–2273, Online. Association for Computational Linguistics.

Fei Li and H. V. Jagadish. 2014. Constructing an interactive natural language interface for relational databases. Proc. VLDB Endow., 8(1):73–84.

Haoyang Li, Jing Zhang, Cuiping Li, and Hong Chen. 2023a. Resdsql: Decoupling schema linking and skeleton parsing for text-to-sql. In Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI).

Jinyang Li, Binyuan Hui, Ge Qu, Binhua Li, Jiaxi Yang, Bowen Li, Bailin Wang, Bowen Qin, Rongyu Cao, Ruiying Geng, Nan Huo, Chenhao Ma, Kevin C. C. Chang, Fei Huang, Reynold Cheng, and Yongbin Li. 2023b. Can llm already serve as a database interface? a big bench for large-scale database grounded textto-sqls.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. pages 311–318.

Mohammadreza Pourreza and Davood Rafiei. 2023. Din-sql: Decomposed in-context learning of text-to-sql with self-correction. arXiv preprint arXiv:2304.11015.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal ofMachine Learning Research, 21(1):5485–5551.

Peng Shi, Patrick Ng, Zhiguo Wang, Henghui Zhu, Alexander Hanbo Li, Jun Wang, Cicero Nogueira dos Santos, and Bing Xiang. 2020a. Learning contextual representations for semantic parsing with generationaugmented pre-training.

Tianze Shi, Chen Zhao, Jordan Boyd-Graber, Hal Daumé III, and Lillian Lee. 2020b. On the potential of lexico-logical alignments for semantic parsing to SQL queries. In Findings ofthe Association for Computational Linguistics: EMNLP 2020, pages 1849–1864, Online. Association for Computational Linguistics.

Ruoxi Sun, Sercan O Arik, Hootan Nakhost, Hanjun Dai, Rajarishi Sinha, Pengcheng Yin, and Tomas Pfister. 2023. Sql-palm: Improved large language modeladaptation for text-to-sql. arXiv preprint arXiv:2306.00739.

Lappoon R. Tang and Raymond J. Mooney. 2001. Using multiple clause constructors in inductive logic programming for semantic parsing. In Proceedings of the 12th European Conference on Machine Learning, pages 466–477, Freiburg, Germany.

Ping Wang, Tian Shi, and Chandan K. Reddy. 2020. Text-to-sql generation for question answering on electronic medical records. In Proceedings of The Web Conference 2020, WWW ’20, page 350–361, New York, NY, USA. Association for Computing Machinery.

Navid Yaghmazadeh, Yuepeng Wang, Isil Dillig, and Thomas Dillig. 2017. Sqlizer: Query synthesis from natural language. Proc. ACM Program. Lang., 1(OOPSLA).

Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. 2018. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task.

Victor Zhong, Caiming Xiong, and Richard Socher. 2017. Seq2sql: Generating structured queries from natural language using reinforcement learning. CoRR, abs/1709.00103.

## Appendix

## A Dataset Details

• Table 8 shows master transaction table.

• Table 9 shows Chart of account table.

• Table 10 shows Customer table

• Table 11 shows Vendor table.

• Table 12 shows Employee table.

• Table 13 shows Product Service table.

• Table 14 shows Payment Methods table.

Distribution of Businesses in BookSQL The distributions of all businesses and their products are shown in Figure 4. Each industry is represented in the inner circle layer, which is followed by its businesses (in the middle circle) and its products in the outer circle. Each industry comprises multiple businesses, and each business consists of multiple products and services.

![](images/2632851d0421c9b6209fc2ae8533d6effec7a02ea1b0821aa80dd947811fa3bb.jpg)  
Figure 3: Sample BookSQL Business Distribution. The middle section shows the sample set of businesses, inner section shows the industries associated with the corresponding business and outer most section shows the corresponding product of the business. This chart is made with the information available at: https://www. ibisworld.com/united-states/list-of-industries/.

![](images/6167e399fde7e2d8cfee052546f2598a7af948a49319e64a8f300f13dc12c110.jpg)  
Figure 4: BookSQLBusiness Distribution. Here, inner circle indicates the industries , middle circle shows the sets of businesses associated to respective industry , and the outer most circle indicate corresponding product of the business. This chart is made with the information available at: https://www.ibisworld.com/united-states/ list-of-industries/.

## B Dynamic Few-shot Prompt + GPT4

Pseudo-code for dynamic few-shot train example selection for a given test question:

```python
from l a n g c h a i n . embeddings . h u g g i n g fa c e import HuggingFaceEmbeddings
from l a n g c h a i n . p r o m p t s . e x a m p l e _ s e l e c t o r import M a x M a r g i n a l R e l e v a n c e E x a m p l e S e l e c t o r
from l a n g c h a i n . v e c t o r s t o r e s import Chroma
e x a m p l e _ s e l e c t o r =
M a x M a r g i n a l R e l e v a n c e E x a m p l e S e l e c t o r . fro m _ e x a m p l e s (
examples ,
HuggingFaceEmbeddings ( model_name=" a l l −MiniLM−L6−v2 " ) ,
Chroma ,
k =10 ,
i n p u t _ k e y s = [ " i n p u t " ]
)
```

## B.1 Example Prompt

Database schema:

Table master\_txn\_table, columns = [\*, Transaction\_ID, Transaction\_DATE, Transaction\_TYPE, Amount,   
CreatedDATE, CreatedUSER, Account, AR\_paid, AP\_paid, Due\_DATE, Open\_balance, Customers,   
Vendor, Product\_Service, Quantity, Rate, Credit, Debit, payment\_method, Misc]   
Table chart\_of\_accounts, columns = [\*, Account\_name, Account\_type] Table customers, columns = [\*,   
customer\_name, customer\_full\_name, Billing\_address, Billing\_city, Billing\_state, Billing\_ZIP\_code,   
Shipping\_address, Shipping\_city, Shipping\_state, Shipping\_ZIP\_code, Balance]   
Table employees, columns = [\*, Employee\_name, Employee\_ID, Hire\_date, Billing\_rate, Deleted]   
Table products, columns = [\*, Product\_Service, Product\_Service\_type]   
Table vendors, columns = [\*, Vendor\_name, Billing\_address, Billing\_city, Billing\_state, Billing\_ZIP\_code,   
Balance]   
Table payment\_method, columns = [\*, Payment\_method, Credit\_card]

Foreign\_keys = [master\_txn\_table.Account = chart\_of\_accounts.Account\_name, master\_txn\_table.Customers = customers.customer\_name, master\_txn\_table.Vendor = vendors.Vendor\_name, master\_txn\_table.Product\_Service = products.Product\_Service, master\_txn\_table.payment\_method = payment\_method.payment\_method]

Following are the example of questions and corresponding SQL queries. «10 Few shot examples from train set»

Translate following question to SQL query.

Input: How much open credit does customer Ronald Bailey have?

Output: SELECT

## C DIN-SQL+GPT4 Prompts

Following section shows the sample prompts used in different DIN-SQL modules. For brevity, we have added only 1 few shot example in these sample prompts. Though in practice, 5-10 few shot examples are used and is mentioned at the end of prompt in « ».

## C.1 Schema Linking Prompt

Table master\_txn\_table, columns = [\*, Transaction\_ID, Transaction\_DATE, Transaction\_TYPE, Amount, CreatedDATE, CreatedUSER, Account, AR\_paid, AP\_paid, Due\_DATE, Open\_balance, Customers, Vendor, Product\_Service, Quantity, Rate, Credit, Debit, payment\_method, Misc]

Table chart\_of\_accounts, columns = [\*, Account\_name, Account\_type]

Table customers, columns = [\*, customer\_name, customer\_full\_name, Billing\_address, Billing\_city,

Billing\_state, Billing\_ZIP\_code, Shipping\_address, Shipping\_city, Shipping\_state, Shipping\_ZIP\_code, Balance]

Table employees, columns = [\*, Employee\_name, Employee\_ID, Hire\_date, Billing\_rate, Deleted] Table products, columns = [\*, Product\_Service, Product\_Service\_type]

Table vendors, columns = [\*, Vendor\_name, Billing\_address, Billing\_city, Billing\_state, Billing\_ZIP\_code, Balance]

Table payment\_method, columns = [\*, Payment\_method, Credit\_card]

Foreign\_keys = [master\_txn\_table.Account = chart\_of\_accounts.Account\_name, master\_txn\_table.Customers = customers.customer\_name, master\_txn\_table.Vendor = vendors.Vendor\_name, master\_txn\_table.Product\_Service = products.Product\_Service, master\_txn\_table.payment\_method = payment\_method.payment\_method]

Q: How much open credit does customer Ronald Bailey have?

S: select sum(open\_balance) from ( select distinct transaction\_id, open\_balance from master\_txn\_table where customers = ’Ronald Bailey’)

A: Let’s think step by step. In the question "How much open credit does customer Ronald Bailey?", we are asked: "How much open credit", so we need column = [master\_txn\_table.open\_balance] "open credit does customer Ronald Bailey", so we need column = [master\_txn\_table.transaction\_id, master\_txn\_table.customers] Based on the columns and tables, we need these Foreign\_keys = []. Based on the tables, columns, and Foreign\_keys, The set of possible cell values are = [Ronald Bailey]. So the Schema\_links are: Schema\_links: [master\_txn\_table.open\_balance, master\_txn\_table.customers, master\_txn\_table.transaction\_id, Ronald Bailey]

«9 MORE FEW-SHOT EXAMPLES»

## C.2 Classification prompt

Q: What are my transactions MTD?

schema\_links: [master\_txn\_table.transaction\_id, master\_txn\_table.amount, master\_txn\_table.transaction\_date]

A: Let’s think step by step. The SQL query for the question "What are my transactions MTD?" needs these tables = [master\_txn\_table], so we don’t need JOIN. Plus, it doesn’t require nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""]. So, we don’t need JOIN and don’t need nested queries, then the the SQL query can be classified as "EASY". Label: "EASY"

Q: How many products are never sold with total value higher than 5?

schema\_links: [Product\_Service.transaction\_id, master\_txn\_table.transaction\_type]

A: Let’s think step by step. The SQL query for the question "How many products are never sold with total value higher than 5?" needs these tables = [Product\_Service,master\_txn\_table], so we need JOIN. Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN) or inner query inside from clause, and we need the answer to the questions = ["products that are sold with total value higher than 5"]. So, we need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".

Label: "NESTED"

## Q: YTD, what was our smallest expense?

schema\_links = [master\_txn\_table.account = chart\_of\_accounts.account\_name, master\_txn\_table.credit, master\_txn\_table.transaction\_date, master\_txn\_table.account\_type, master\_txn\_table.debit]

A: Let’s think step by step. The SQL query for the question "YTD, what was our smallest expense?" needs these tables = [master\_txn\_table,chart\_of\_accounts], so we need JOIN. Plus, it doesn’t need nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""]. So, we need JOIN and don’t need nested queries, then the the SQL query can be classified as "NON-NESTED".

Label: "NON-NESTED"

## C.3 SQL Generation

## C.3.1 Easy Prompt

Q: "How much open credit does customer Ronald Bailey?"

Schema\_links: [master\_txn\_table.open\_balance, master\_txn\_table.transaction\_id, master\_txn\_table.customers,Ronald Bailey]

SQL: select sum(open\_balance) from ( select distinct transaction\_id, open\_balance from master\_txn\_table where customers = ’Ronald Bailey’)

«4 MORE FEW-SHOT EXAMPLES»

## C.3.2 Non-Nested Complex Prompt

Q: "How many Traveller accomodation did we sell to Ethan Walker today?"

Schema\_links: [master\_txn\_table.quantity ,master\_txn\_table.customers, master\_txn\_table.product\_service, master\_txn\_table.transaction\_type, master\_txn\_table.transaction\_date] A: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = []. First, create an intermediate representation, then use it to construct the SQL query. Intermediate\_representation: select sum(master\_txn\_table.quantity) from master\_txn\_table where master\_txn\_table.customers = ’Ethan Walker’ and master\_txn\_table.product\_service = ’Traveller accomodation’ and master\_txn\_table.trasaction\_type in (’invoice’,’sales receipt’) and master\_txn\_table.transaction\_date BETWEEN date(current\_date) AND date(current\_date) SQL: select sum(quantity) from master\_txn\_table where customers = Ëthan Walkeränd product\_service = Traveller accomodationänd trasaction\_type in (’invoice’,’sales receipt’) and transaction\_date BETWEEN<sup>¨</sup> date(current\_date) AND date(current\_date)

«9 MORE FEW-SHOT EXAMPLES»

## C.3.3 Nested Complex Prompt

Q: "How many products are never sold with total value higher than 5?" Schema\_links: [master\_txn\_table.product\_service, master\_txn\_table.transaction\_type, master\_txn\_table.credit, product\_service.\*]

A: Let’s think step by step. "How many products are never sold with total value higher than 5?" can be solved by knowing the answer to the following sub-question "Show me all the products which are never sold with total credit value higher than 5?". The SQL query for the sub-question "Show me all the products which are never sold with total credit value higher than 5?" is SELECT count(\*) FROM Product\_Service WHERE product\_service NOT IN ( SELECT product\_service FROM master\_txn\_table WHERE transaction\_type in (’invoice’,’sales receipt’) group by product\_service having sum(credit)>5) So, the answer to the question "How many products are never sold with total value higher than 5?" is = Intermediate\_representation: SELECT count(Product\_Service.\*) FROM Product\_Service WHERE Product\_Service.product\_service NOT IN ( SELECT master\_txn\_table.product\_service FROM mas ter\_txn\_table WHERE master\_txn\_table.transaction\_type in (’invoice’,’sales receipt’) group by master\_txn\_table.product\_service having sum(master\_txn\_table.credit) > 5)

SQL: SELECT count(\*) FROM Product\_Service WHERE product\_service NOT IN ( SELECT product\_service FROM master\_txn\_table WHERE transaction\_type in (’invoice’,’sales receipt’) group by product\_service having sum(credit) > 5)

«9 MORE FEW-SHOT EXAMPLES»

## C.4 Self Correction Prompt

For the given question, use the provided tables, columns, foreign keys, and primary keys to fix the given SQLite SQL QUERY for any issues. If there are any problems, fix them. If there are no issues, return the SQLite SQL QUERY as is.

Use the following instructions for fixing the SQL QUERY:

1) Use the database values that are explicitly mentioned in the question.

2) Pay attention to the columns that are used for the JOIN by using the Foreign\_keys.

3) Use DESC and DISTINCT when needed.

4) Pay attention to the columns that are used for the GROUP BY statement.

5) Pay attention to the columns that are used for the SELECT statement.

6) Only change the GROUP BY clause when necessary (Avoid redundant columns in GROUP BY).

7) Use GROUP BY on one column only.

## D Evaluation Metrics

The following standard metrics are used:

• Exact Match Accuracy (Yu et al., 2018): Both predicted and the Gold SQL are decomposed into different SQL components like SELECT, WHERE, GROUP BY, etc. Predicted SQL is marked as correct if all SQL components exactly match with the Gold SQL.

• Execution Accuracy (Yu et al., 2018): Output of predicted SQL is the same as Gold SQL’s output on execution against the database.

• Partial Component Match F1 (Hazoom et al., 2021a): Both the predicted query and the gold query are parsed into tress using JSqlParser<sup>12</sup>. These two parsed trees are compared, and an aggregated score is calculated based on the number of matching sub-trees.

• BLEU-4 (Papineni et al., 2002): It measures the number of matching n-grams between the predicted and the Gold SQL.

• ROUGE-L (Lin, 2004): It is based on the longest common sub-sequence (LCS) between the predicted and the Gold SQL. A longer shared sequence indicates more similarity between the predicted and the Gold SQL.

## E Training Details and Hyper-parameters

All experiments were done on a single NVIDIA A10G Tensor Core GPU.

For SEDE, we used T5-Large as the base seq-to-seq model, with a learning rate of 5e  5 with 15 epochs and batch size of 6. For decoding, a beam size of 6 was used, with max decoding steps of 250.

For UniSAr, we use T5-Large as a base language model with a learning rate of 1e-5 and max tokens is 1024. We adopt the polynomial\_decay with 5,000 warmup updates. The dropout rate is 0.1. Optimizer is Adam with the default parameters. The max-update is set to 10,000. Empirically, the model obtained the best performance about 10 15 epochs in BookSQL. The Fairseq dynamically tunes the batch size to realize higher GPU utilization.

For RESDSQL, we used settings recommended by the original paper and code. The Schema Item Classifier module used a RoBERTa-large model with a learning rate of 1e  5 and an effective batch size of 32 (using gradient accumulation). topk\_table\_num value of 4 and topk\_column\_num value of 8 were used. For the text2sql module, a T5-large model was used with a learning rate of 5e  5 and an effective batch size of 32 (using gradient accumulation). Beam search decoding was used with num\_beams set to 8 and num\_return\_sequences set to 8.

For DIN-SQL+GPT4 and Dynamic few shot prompt + GPT4, we used OpenAI GPT4 API with following settings: n = 1, temperature=0.0, max\_tokens=600, top\_p = 1.0, frequency\_penalty=0.0, presence\_penalty=0.0. Given limitations on the number of calls to OpenAI GPT4 API, we used a random 10% of BookSQL test set for GPT4-based approaches.

<table><tr><td>busin- ess Id</td><td>Trans- action ID</td><td>Trans- action date</td><td>Trans- action type</td><td>Amount Creat- ed date</td><td>ed</td><td>Creat- user</td><td>Acco- unt</td><td>A/R paid</td><td>A/P paid</td><td>Due date</td><td>Open bal- ance</td><td>Cust- omer name</td><td>Vendor name uct vice</td><td>Prod- Ser-</td><td>Quan- tity</td><td>Rate</td><td>Credit</td><td>Debit</td></tr><tr><td>4</td><td>1867</td><td>2022- 08-31</td><td>credit card credit</td><td>999.58</td><td>2023- 05-11</td><td>Joshua Hud- son</td><td>Visa</td><td></td><td></td><td>2023- 09-17</td><td>232.85</td><td></td><td></td><td></td><td></td><td></td><td></td><td>999.58</td></tr><tr><td>4</td><td>1867</td><td>2022- 08-31</td><td>credit card</td><td>999.58</td><td>2023- 05-11</td><td>Joshua Hud- son</td><td>Savings -</td><td></td><td></td><td>2023- 09-17</td><td>232.85</td><td></td><td></td><td></td><td></td><td></td><td>999.58</td><td></td></tr><tr><td>4</td><td>1716</td><td>2022- 08-15</td><td>Bill</td><td>784.19</td><td>2023- 05-14</td><td>Joshua Hud- son</td><td>Accounts - Payable (A/P)</td><td></td><td>unpaid</td><td>2023- 07-12</td><td>539.03</td><td></td><td>Jade Bar- nett</td><td></td><td></td><td></td><td>784.19</td><td></td></tr><tr><td>4</td><td>1716</td><td>2022- 08-15</td><td>Bill</td><td>784.19</td><td>2023- 05-14</td><td>Joshua Hud- son</td><td>Lawyer</td><td></td><td>unpaid</td><td>2023- 07-12</td><td>539.03</td><td>Jade</td><td>Bar- nett</td><td></td><td></td><td></td><td></td><td>784.19</td></tr><tr><td>4</td><td>1818</td><td>2022- 08-17</td><td>Payment 2204</td><td></td><td>2022- 11-22</td><td>Joshua Hud- son</td><td>prepaid ex- penses</td><td>paid</td><td></td><td>2023- 06-25</td><td>1841.82</td><td>Andrew Rose</td><td></td><td></td><td></td><td></td><td></td><td>2204</td></tr><tr><td>4</td><td>1818</td><td>2022- 08-17</td><td>Payment 2204</td><td></td><td>2022- 11-22</td><td>Joshua Hud- son</td><td>Accounts paid Re- ceiv-</td><td></td><td>2023- 06-25</td><td></td><td>1841.82</td><td>Andrew Rose</td><td></td><td></td><td></td><td>2204</td><td></td><td></td></tr></table>

Table 8: Master Transaction Table

<table><tr><td>Business Id</td><td>Account name</td><td>Account Full Name</td><td>Account type</td></tr><tr><td>2</td><td>Accumulated Depreciation</td><td>Accumulated Depreciation</td><td>Fixed Asset</td></tr><tr><td>2</td><td>Furniture and Equipment</td><td>Furniture and Equipment</td><td>Fixed Asset</td></tr><tr><td>2</td><td>Payroll Liabilities</td><td>Payroll Liabilities</td><td>Other Current Liability</td></tr><tr><td>2</td><td>Opening Balance Equity</td><td>Opening Balance Equity</td><td>Equity</td></tr><tr><td>2</td><td>Owners Draw</td><td>Owners Draw</td><td>Equity</td></tr><tr><td>2</td><td>Owners Equity</td><td>Owners Equity</td><td>Equity</td></tr><tr><td>2</td><td>Accounting Service Income</td><td>Accounting Service Income</td><td>Income</td></tr><tr><td></td><td>Consulting Income</td><td>Consulting Income</td><td>Income</td></tr><tr><td>22</td><td>Tax Preparation Services Income</td><td>Tax Preparation Services Income</td><td>Income</td></tr></table>

Table 9: Chart of Account

<table><tr><td>Business Id</td><td>Customer name</td><td>Customer full name</td><td>Billing address</td><td>Billing city</td><td>Billing state</td><td>Billing ZIP code</td><td>Shipping dress</td><td>ad- Shipping city</td><td>Shipping state</td><td>Shipping ZIP code</td><td></td><td>Balance</td></tr><tr><td>2</td><td>Valerie Kline</td><td>Valerie Kline</td><td>5120 Shelia Val- leys Suite 824</td><td>New Cyn- thiaburgh</td><td>AL</td><td>18662</td><td>40109 Pamela Ex- tension</td><td>West Patrickville</td><td>TN</td><td></td><td>21599</td><td>67.32</td></tr><tr><td>2</td><td>Greg Carde- nas</td><td>Greg Carde- nas</td><td>59614 Margaret Roads</td><td>Transide</td><td>OK</td><td>72668</td><td>3758 Savage Gar- den Suite 126</td><td></td><td>Seanbury</td><td>VT</td><td>7317</td><td>167.00</td></tr><tr><td>2</td><td>Mr. Zachary Levy</td><td>Mr. Zachary Levy</td><td>30939 Brandon Ford Suite 571</td><td>South Joan- ntown</td><td>NV</td><td>71097</td><td>2752 Austin Brooks Suite 864</td><td>Stoutville</td><td>MI</td><td></td><td>51839</td><td>69.34</td></tr><tr><td>2</td><td>Taylor Hughes</td><td>Taylor Hughes</td><td>7945 Soto Point</td><td>Monica- mouth</td><td>ND</td><td>46573</td><td>04225 Edwards Valley Suite 176</td><td></td><td>Taylorton</td><td>DE</td><td>516</td><td>169.34</td></tr><tr><td>2</td><td>Jodi Bishop</td><td>Jodi Bishop</td><td>850 Brent Parks</td><td>Shieldsberg</td><td>AL</td><td>7549</td><td>1558 Brown Hills</td><td>South Robert</td><td></td><td></td><td>94105</td><td>799.37</td></tr><tr><td>2</td><td>Andrew Flores</td><td>Andrew Flores</td><td>3141 Jamie Isle Apt. 494</td><td>South Nicholas- mouth</td><td>OH</td><td>71180</td><td>4662 Peters Park- ways Suite 775</td><td></td><td>Desireebury AR</td><td></td><td>34741</td><td>79.37</td></tr><tr><td>2</td><td>Earl Lee</td><td>Earl Lee</td><td>017 Lisa Skyway</td><td>Lake Kristineb- urgh</td><td>AL</td><td>79642</td><td>8285 Thornton Motorway Suite 926</td><td></td><td>Lawsonville ID</td><td></td><td>12448</td><td>84.75</td></tr><tr><td>2</td><td>Thomas Jackson</td><td>Thomas Jackson</td><td>09653 Christian Stravenue</td><td>North John- town</td><td>MS</td><td>43479</td><td>1205 Fork Suite 756</td><td>Shawna</td><td>Tracymouth</td><td>TX</td><td>77146</td><td>684.7</td></tr><tr><td>2</td><td>Jason John- son</td><td>Jason John- son</td><td>46474 Alan Cove Suite 685</td><td>Michaelside</td><td>VT</td><td>50177</td><td>Unit 0702 Box 5832</td><td></td><td>DPO</td><td>AA</td><td>15548</td><td>747.44</td></tr><tr><td>2</td><td>Craig Greer</td><td>Craig Greer</td><td>496 Moreno Brooks</td><td>Lake Katri- namouth</td><td>NM</td><td>14367</td><td>USNV Gutierrez</td><td></td><td>FPO</td><td>AP</td><td>71930</td><td>47.34</td></tr><tr><td>2</td><td>Jeffrey Fisher</td><td>Jeffrey Fisher</td><td>632 Robert Plains Apt. 260</td><td>Woodardville</td><td>LA</td><td>86396</td><td>Apt. 818</td><td>46671 Joseph Flat</td><td>Sweeneyshire DC</td><td></td><td>70691</td><td>5829.13</td></tr></table>

Table 10: Customer Table

<table><tr><td>Business Id</td><td>Vendor name</td><td>Billing address</td><td>Billing city</td><td>Billing state</td><td>Billing ZIP code</td><td>Balance</td></tr><tr><td>2</td><td>Shelly Ramos</td><td>82768 Dawn Cres- cent</td><td>West Cynthia</td><td>WY</td><td>39877</td><td>4042.15</td></tr><tr><td>2</td><td>Jade Barnett</td><td>782 Mitchell Camp Suite 676</td><td>Grahambury</td><td>KS</td><td>80370</td><td>12949.89</td></tr><tr><td>2</td><td>Nicole Jordan</td><td>14959 Mccullough</td><td>East Kevinfurt</td><td>WI</td><td>42930</td><td>5294.89</td></tr><tr><td>2</td><td>Adam Pena</td><td>Green Suite 029 192 Brenda Gardens</td><td>Erinmouth</td><td>IA</td><td>93008</td><td>6949.89</td></tr><tr><td>2</td><td>Jeffrey Roman</td><td>784 Cameron Parks Apt. 902</td><td>North Gloriafurt</td><td>AR</td><td>48141</td><td>7299.89</td></tr><tr><td>2</td><td>Zachary Butler</td><td>61717 Christopher Cliffs Apt. 122</td><td>Port Joshua</td><td>MT</td><td>44164</td><td>465.09</td></tr><tr><td>2</td><td>Taylor Moses</td><td>19368 Jenny Courts</td><td>Kerristad</td><td>OR</td><td>25430</td><td>65.09</td></tr><tr><td>2</td><td>John Russo</td><td>Apt. 094 Unit 6387 Box 0856</td><td>DPO</td><td>AA</td><td>73133</td><td>1538.8</td></tr><tr><td>2</td><td>Robert Phillips</td><td>USCGC Steele</td><td>FPO</td><td>AA</td><td>91533</td><td>55388.8</td></tr></table>

Table 11: Vendor Table

<table><tr><td>Business Id</td><td>Employee name</td><td>Employee ID</td><td>Hire date</td><td>Billing rate</td><td>Deleted</td></tr><tr><td>2</td><td>Stephanie Baker</td><td>STE123</td><td>07/17/2022</td><td></td><td>No</td></tr><tr><td>2</td><td>Julia Rivera</td><td>JUL456</td><td>07/31/2002</td><td></td><td>No</td></tr><tr><td>2</td><td>Valerie Kline</td><td>VAL232</td><td>04/15/2012</td><td></td><td>Yes</td></tr><tr><td>2</td><td>Greg Cardenas</td><td>GRE443</td><td>08/27/2013</td><td>一</td><td>No</td></tr><tr><td>2</td><td>Mr. Zachary Levy</td><td>ZAC998</td><td>01/28/2000</td><td></td><td>Yes</td></tr><tr><td>2</td><td>Taylor Hughes</td><td>TAY009</td><td>07/17/2022</td><td></td><td>Yes</td></tr><tr><td>2</td><td>Jodi Bishop</td><td>JOD778</td><td>12/27/2016</td><td></td><td>Yes</td></tr><tr><td>2</td><td>Andrew Flores</td><td>AND667</td><td>05/20/2018</td><td>一</td><td>No</td></tr><tr><td>2</td><td>Earl Lee</td><td>EAR221</td><td>08/19/2002</td><td>一</td><td>No</td></tr></table>

Table 12: Employee Tables

<table><tr><td>Business Id</td><td>Product_service</td><td>Product_Service_type</td></tr><tr><td>2</td><td>Hours</td><td>Service</td></tr><tr><td>2</td><td>Services</td><td>Service</td></tr><tr><td>2</td><td>Design</td><td>Service</td></tr><tr><td>2</td><td>Installation</td><td>Service</td></tr><tr><td>2</td><td>Lighting</td><td>Service</td></tr><tr><td>2</td><td>Maintenance &amp; Repair</td><td>Service</td></tr><tr><td>2</td><td>Refunds &amp; Allowances</td><td>Service</td></tr></table>

Table 13: Product Service Table

<table><tr><td>Business Id</td><td>Payment method</td><td>Credit card</td></tr><tr><td>1</td><td>Cash</td><td>No</td></tr><tr><td>1</td><td>Check</td><td>No</td></tr><tr><td>1</td><td>Visa</td><td>Yes</td></tr><tr><td>1</td><td>MasterCard</td><td>Yes</td></tr><tr><td>1</td><td>Discover</td><td>Yes</td></tr><tr><td>1</td><td>American Express</td><td>Yes</td></tr><tr><td>1</td><td>Diners Club</td><td>Yes</td></tr></table>

Table 14: Payment Methods