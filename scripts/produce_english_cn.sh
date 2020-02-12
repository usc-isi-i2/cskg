cat data/conceptnet-assertions-5.7.0.csv | awk '$2!~/^\/r\/dbpedia\// && $3~/^\/c\/en\// && $4~/^\/c\/en\//' > data/conceptnet-en.csv
#cat data/conceptnet-assertions-5.7.0.csv | awk '$2!~/^\/r\/dbpedia\// && $3~/^\/c\/en\// && ($4~/^\/c\/en\// || $2=="/r/ExternalURL")' > data/conceptnet-en-with-externalurl.csv
