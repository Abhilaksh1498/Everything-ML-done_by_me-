SELECT * FROM web_events;         -- These are used for single line comments
/* Multi-line comments
1. commands in sql are case-insensitive 
	sElEcT == select == SELECT
2. Whitespaces are ignored
	SELECT * 
	FROM web_events will run same as above
3. Semi colons are not "necessary", but its a good practice. Also if you are using more than 1 statements, you'll need ;
4. * is used to select all cols
*/
SELECT id, account_name FROM web_events;      -- to select specific cols
LIMIT 10;                                     -- LIMITS THE NO OF ROWS TO THAT NUMBER
											  -- it will always come at the last in any statement

/* ORDER BY command is used to sort according to a specific col, comes AFTER from command but before limit
By default its in ascending order A-Z, 1-10 or in case of time, earliest to latest etc.
To sort in descending order, use DESC keyword */

SELECT * FROM web_events
ORDER BY occurred_at DESC       				-- DESC stands for DESCending
LIMIT 20;

/* Ordering by multiple cols 
Starts from left to right, ordering first happens on leftmost then on that obtained subset with the next col and so on*/
SELECT id, account_id, total_amt_usd
FROM orders
ORDER BY account_id DESC, total_amt_usd;

/* WHERE command is used to filter based on column values*/
SELECT * FROM  orders
WHERE gloss_amt_usd >= 1000                    -- other filters are =, >,<, !=, <=, >=
ORDER BY total DESC, poster_qty
LIMIT 5;

/* Using WHERE's = & != for non-numeric data */
SELECT primary_poc, name, website FROM accounts
WHERE name= 'Exxon Mobil'                      -- SQL requires single-quotes, not double-quotes, around text values

/* Derived Columns
These are temporary cols that can be created during query that hold manipulation of certain cols present in database*/
SELECT id, account_id, 
   poster_amt_usd/(standard_amt_usd + gloss_amt_usd + poster_amt_usd) AS post_per        -- displays a new temporary "post_per" col
  																						-- with values governed by that mathematical rule
FROM orders
LIMIT 10;

/* LOGICcal OPERATORS*/
-- LIKE its similar to find function in python. It checks where a specific set of characters are present or not
SELECT * FROM accounts
WHERE name LIKE '%s'                            -- All those rows for whom the name ends with alphabet 's'
												-- % sign is used to indicate any no of characters (including whitespace)
												-- on either side
/* IN keyword */ 
SELECT * FROM web_events
WHERE channel IN ('organic', 'adwords')			-- does the job of channel = a or b or c ...
												-- can be numbers too in that case no '' ofc 
												-- always need curly braces even if using only 1 word

/* NOT keyword - performs the inverse, can be used in LIKE, IN */												
SELECT *
FROM web_events
WHERE channel NOT IN ('organic', 'adwords');

/* The AND operator is used within a WHERE statement to consider more than one
logical clause at a time. 
Each time you link a new statement with an AND, you will need to specify the column you are interested in looking at. 
This works with all of the operations we have seen so far including arithmetic operators (+, *, -, /). LIKE, IN, and NOT logic. */
SELECT * FROM web_events
WHERE (channel IN ('organic', 'adwords')) AND (occurred_at BETWEEN '2016-01-01' AND '2016-12-31') OR website LIKE '%google%'
ORDER BY occurred_at DESC;                    -- between is inclusive of endpoints between a and b => [a,b]
											  -- timestamp data is compared in quotes

/*SQL JOIN - used for working with >1 tables with information for cells of particular col residing in a different table*/
SELECT orders.account_id, orders.total, accounts.* FROM orders        -- this displays 2 coulmns from orders, all cols from accounts
JOIN accounts 														  -- You could've used accounts in FROM and orders in JOIN as well
ON orders.account_id = accounts.account_id							  -- which side of the = a column is listed doesn't matter
LIMIT 10;

/* Primary Key (PK)
This col has Unique val per row. Usually the first col */
/* Foreign Key (FK)
This col is a PK in another table with the help of which we could join two tables.
they are associated with the crow-foot notation above to show they can appear multiple times in a particular table*/

/* JOINING MULTIPLE TABLES */
SELECT web_events.channel, accounts.name, orders.total
FROM web_events
JOIN accounts											-- Join holds a table, and ON is a link for our PK to equal the FK.
ON web_events.account_id = accounts.id
JOIN orders
ON accounts.id = orders.account_id

/* Alias
Used for referencing table or column names by shorter/more readable names */
SELECT *
FROM tablename AS t1
JOIN tablename2 AS t2

-- You could skip AS clause and the results will still remain the same, so the following also works
SELECT *
FROM tablename t1
JOIN tablename2 t2 
ON t1.col1 = t2.col2

-- Now you could use use t1 and t2 for tablenames anwhere in the query
SELECT t1.col3, t1.col4, t2.col5
FROM tablename t1
JOIN tablename2 t2 
ON t1.col1 = t2.col2

/* ALIASING IN COLUMN NAMES */
SELECT column2 as c2 FROM web_events -- OR
SELECT column2 c2, column3 c3 FROM web_events			-- Note however, during query the column name displayed will be c2/c3

SELECT r.name r_name, s.name rep
FROM sales_reps 										-- if we have a common col name in two tables, and we want to displa 
JOIN region 											-- both cols, then we need to alias (atleast) one of those cols
ON s.region_id = r.id

/* Note that what were doing above was "inner" joins
Consider a two circle venn diagram. The left circle is table in FROM clause
The right circle is table in JOIN clause. Only those rows appear in final result for which ON clause is true in both of them --
That's like the intersection. 

"outer join -- This is like taking A\/(A/\B) or (A/\B)\/B depending on LEFT & RIGHT JOIN respectively
OUTER JOIN takes A\/B. Also, LEFT JOIN = LEFT OUTER JOIN

As left & right joins are interchangable we mostly use left joins.
*/

-- Logic in where clause occurs AFTER the join clause
SELECT s.name AS sales_rep_name, r.name as region_name, a.name AS account_name
FROM region AS r
JOIN sales_reps s
ON r.id = s.region_id
JOIN accounts AS a
ON a.sales_rep_id = s.id
WHERE r.name = 'Midwest'              -- has to come before order by and after any joins
ORDER BY account_name

/* Where keyword filters at the very end irrespective of whether its left, right .... or any other join
and the keyword AND, when inserted inside ON statement, has to be considered like a filtered version of the table before using join
Its basically the filtering done before join --> its like pehle subset bana lo based on and condition, fir join karlo, and 
then agar left ya right join h to jo bhi bacha h usko chipka lo*/

select a.name as account_name, r.name as region_name, o.total_amt_usd/(o.total+.01) as unit_price from orders o
join accounts a
on o.account_id = a.id
 --- if i write the where statement here, I get an error
join sales_reps s
on s.id = a.sales_rep_id
join region r
on r.id = s.region_id
where o.standard_qty > 100

-- The BETWEEN operator is inclusive: begin and end values are included.

/* NULLS -- cells that have no data. Its not a value so you CAN'T use = operator */
SELECT * FROM web_events
WHERE primary_poc is NULL            -- is NOT NULL

/* Aggregating data -- min/max, mean, sum */
SELECT COUNT(*) AS row_count         -- returns count of all rows having some non null data (i.e. non null data in atleast 1 col)
FROM accounts

-- whereas the below will only return count of all rows having non null data in a specified col
SELECT COUNT(accounts.primary_poc) AS primary_poc_count
from accounts

-- SUM        sum along a **COL**, SUM(*) can't be used, treats null as 0
select sum(poster_qty) s_p, sum(standard_qty) s_s, sum(total_amt_usd) s_t from orders

select sum(standard_qty) from orders
where account_id = 1001

-- you could involve these aggregation functions in mathematical operations
SELECT SUM(standard_amt_usd)/SUM(standard_qty) AS standard_price_per_unit
FROM orders;

SELECT AVG(standard_qty) avg_sq,       -- doesn't take into account the null rows in either numerator/denominator
AVG(gloss_qty) avg_gq,
AVG(poster_qty) avg_pq,
AVG(standard_amt_usd) avg_s_usd,
AVG(gloss_amt_usd) avg_q_usd,
AVG(poster_amt_usd) avg_p_usd from orders

SELECT MAX(occurred_at)
FROM web_events;

/* GROUP BY                     
Always goes btw WHERE & ORDER BY clauses if they are present
used in conjunction with aggregation function to aggregate on 'groups' of data eg. avg_amt_usd by each account_id */
select a.name ,sum(o.total_amt_usd) total_amt_usd from orders o
join accounts a
on a.id = o.account_id
where total_amt_usd > 10000
group by a.name               -- must have the same column as used in the select statement
order by a.name               

/* Intuitive technique for writing GROUP BY statements is
--> REmove the aggregate function in select statement while still keeping that col in select statement 
--> think how would the table appear like after joining
--> Now visualise how would that particular col 'collapse' after applying aggregate function
--> Now think how would the collapsing be like if we group based on the other col */

/* GROUPING BY MULTIPLE COLUMNS */      -- i.e. >1 cols apart from grouping col
select s.name, w.channel, count(w.channel) num_occurances from accounts a
JOIN web_events w
ON w.account_id = a.id
JOIN sales_reps s
ON s.id = a.sales_rep_id
GROUP BY s.name, w.channel                  -- the order of both grouping/ordering is from left to right
ORDER BY s.name, num_occurances DESC

/* DISTINCT 
used to select distinct values in a col 
and can be used as a replacement of GROUP BY clause when we DO NOT want to apply aggregations */    
SELECT DISTINCT s.name, a.id FROM accounts a       -- SELECT s.name, DISTINCT col_xyz is an error 
JOIN sales_reps                                    -- distinct always appears with select
ON s.id = a.sales_rep_id
ORDER by s.name
-- distinct lists the unique (col1, col2, col3 ...) tuples appearing so even though in syntax we write it once
-- its choosing distinct tuples over all the columns that we wish to present (i.e. btw SELECT and FROM keywords)

/* HAVING */
-- Functions like WHERE clause but on aggregated cols
-- WHERE appears before group by/order by; HAVING appears in between GROUP BY and ORDER BY
select s.name, count(a.name) num_acc_managed from sales_reps s
JOIN accounts a
ON a.sales_rep_id = s.id

group by s.name
HAVING count(a.name) > 5           -- note we can't use alias col name as well
order by num_acc_managed

select a.id, w.channel, count(w.channel) from web_events w
JOIN accounts a
ON w.account_id = a.id
-- WHERE w.channel = 'facebook'          would've also returned same result
group by a.id, w.channel
HAVING count(w.channel) > 6 AND w.channel = 'facebook'    -- could also use non aggregated cols 
order by a.id, w.channel

/* Working with dates */
SELECT DATE_PART('year' , o.occurred_at) yyyy, DATE_PART('month', o.occurred_at) mm, sum(o.gloss_amt_usd) from orders o
JOIN accounts a
ON o.account_id = a.id
WHERE a.name = 'Walmart'
GROUP BY 1, 2
ORDER BY 3 DESC            -- the nos. 1,2 represent cols/aggregated cols in select statement in same order
/* other arguments supported with DATE_PART are
century
decade
year
month
day
hour
minute
second
microseconds
milliseconds
dow
doy  ... etc */

/* CASE Statements
Method to generate "derived" cols using WHEN-THEN-ELSE (optional) keywords
Always goes with the select statement
You can make any conditional statement using like WHERE between WHEN and THEN, AND and OR.
Always terminated with END keyword */
select *, 
CASE WHEN account_id >= 4000 THEN 'greater than 4k'       -- multiple WHEN are evaluated from top to bottom
WHEN account_id >= 3000 THEN 'greater than 3k'            -- if ELSE were not there, the values are null for those cells
ELSE 'fuck off' END AS random_shit 
from orders
order by account_id desc
limit 1000

--######## EXAMPLE WHEN WE NEED TO USE DERIVED COLUMN INSIDE OTHER CONDITIONING ##############
-- YOU CANT USE ALIAS NAME as it will say col_name doesn't exist 
select a.name, 
sum(o.total_amt_usd) as total_sales,
CASE WHEN sum(o.total_amt_usd) > 200000 THEN 'first level'
WHEN sum(o.total_amt_usd) between 100000 and 200000 then 'second level'
ELSE 'third level' END AS level 
from accounts a
JOIN orders o
ON a.id = o.account_id
group by 1
order by 2 desc, a.name

/* SUBQUERIES          
--> help you to run queries over the o/p of another query eg. aggregated col derived from another aggregated col etc.
--> the inner/sub query is treated like a table from the pov of outer query.
--> Therefore you can refer to any derived/alias col names defined in inner query from the outer query 
--> written inside () with an alias name after the paranthesis*/ 
SELECT subquery1.channel, AVG(event_count)        -- select subquery1.* from (...) subquery1
from (select DATE_TRUNC('day', occurred_at), channel, count(*) event_count 
	from web_events
	group by 1,2
	order by 1) subquery1          -- inner query is also called "original query" and is executed first,
GROUP BY 1                        -- make sure it runs on its own
ORDER BY 2 DESC;

 /* if you are only returning a single value, you can use that value in a logical statement 
 like WHERE, HAVING, or could be nested within a CASE statement.
 If you return a col then it could be used with the IN logical statement. */
 SELECT * from orders
WHERE DATE_TRUNC('month',occurred_at) = (select DATE_TRUNC('month',min(occurred_at)) from orders)
ORDER BY occurred_at

SELECT r.name, COUNT(o.total) total_orders
FROM sales_reps s
JOIN accounts a
ON a.sales_rep_id = s.id
JOIN orders o
ON o.account_id = a.id
JOIN region r
ON r.id = s.region_id
GROUP BY r.name
HAVING SUM(o.total_amt_usd) = (    -- notice the brackets, and that you dont need aliasing
      SELECT MAX(total_amt)
      FROM (SELECT r.name region_name, SUM(o.total_amt_usd) total_amt
              FROM sales_reps s
              JOIN accounts a
              ON a.sales_rep_id = s.id
              JOIN orders o
              ON o.account_id = a.id
              JOIN region r
              ON r.id = s.region_id
              GROUP BY r.name) sub);

SELECT a.name act_name, SUM(o.standard_qty) tot_std, SUM(o.total) total --- note that >1 aggregated cols can be there 
                         FROM accounts a                                -- with sigle grpuping
                         JOIN orders o
                         ON o.account_id = a.id
                         GROUP BY 1
                         ORDER BY 2 DESC

/* WITH/CTE (Common table expressions)
--> Used to Store queries in the form of tables at the top
--> This will prevent re-running the query again and again if its, for instance, used as a subquery */
WITH table1 AS (
          SELECT *
          FROM web_events),

     table2 AS (
          SELECT *
          FROM accounts)


SELECT *
FROM table1                            -- you could use it just like any other table
JOIN table2                            -- table1, table2 are the alias names
ON table1.account_id = table2.id;
                         
-- Eg. 
WITH events AS (                         --- AS KEYWORD IS VERY IMPORTANT WHILE ALIASING, you cant skip it
          SELECT DATE_TRUNC('day',occurred_at) AS day, 
                        channel, COUNT(*) as events
          FROM web_events 
          GROUP BY 1,2)

SELECT channel, AVG(events) AS average_events
FROM events
GROUP BY channel
ORDER BY 2 DESC;


WITH t1 AS (SELECT r.name region_name, SUM(o.total_amt_usd) total_amt
   FROM sales_reps s
   JOIN accounts a
   ON a.sales_rep_id = s.id
   JOIN orders o
   ON o.account_id = a.id
   JOIN region r
   ON r.id = s.region_id
   GROUP BY r.name
   ORDER by 2 desc
   LIMIT 1)
   
  
  SELECT r.name, count(o.total) total_orders
   FROM sales_reps s
   JOIN accounts a
   ON a.sales_rep_id = s.id
   JOIN orders o
   ON o.account_id = a.id
   JOIN region r
   ON r.id = s.region_id        
   GROUP BY 1 HAVing r.name = (SELECT region_name from t1)            -- we could use CTE without select statement too :)
