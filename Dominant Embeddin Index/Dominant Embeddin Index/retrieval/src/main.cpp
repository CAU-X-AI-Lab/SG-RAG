#include "./rtree/rtree.h"
#include "./rtree/rtnode.h"
#include "./rtree/entry.h"
#include "./blockfile/blk_file.h"
#include "./blockfile/cache.h"
#include "./linlist/linlist.h"
#include "./rtree/rtree_cmd.h"
#include "rand.h"
#include "cdf.h"

#include "./graph/graph.h"
#include "custom.h"

#define NOMINMAX
#undef min
#undef max

#include <cmath>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <string>
#include <map>
#include <omp.h>
#include <chrono>
#include <cstdio>

using namespace std;
using namespace chrono;

string dataset_path = "D:\\test-code\\GNN-PE-main\\GNN-PE-main\\online\\Datasets\\yeast\\";

int embedding_dimension = 2;
int partition_number = 5;
int GNN_number = 3;
int path_length = 2;
int path_nodes = path_length + 1;
int query_number = 100;

int main()
{
	string filename = dataset_path + "data_graph.txt";
	Graph *data_graph = new Graph(true);
	data_graph->loadGraphFromFile(filename);
	data_graph->printGraphMetaData();

	Partition **partitions = new Partition *[partition_number];
	for (int i = 0; i < partition_number; i++)
	{
		filename = dataset_path + "Partition-" + to_string(i) + "/";
		partitions[i] = new Partition(filename, GNN_number, query_number, data_graph);

		cout << endl
			 << "Partition " << to_string(i) << ":" << endl;

		partitions[i]->build_index();
	}

	filename = dataset_path + "query_graphs/";
	Query_Graph **query_graphs = new Query_Graph *[query_number];
	double query_time = 0;
	for (int query_ID = 0; query_ID < query_number; query_ID++)
	{
		query_graphs[query_ID] = new Query_Graph(filename, query_ID);

		double query_plan_time = query_graphs[query_ID]->generate_query_plan_2();

		double search_time = 0;

		vector<vector<Candidate_Path>> global_candidates(query_graphs[query_ID]->query_plan.size());
		vector<vector<int>> global_boundary(query_graphs[query_ID]->query_plan.size());

		vector<vector<vector<int>>> candidates(partition_number);

#pragma omg parallel for num_threads(partition_number)
		for (int i = 0; i < partition_number; i++)
		{
			search_time += partitions[i]->query(query_graphs[query_ID], query_ID, candidates[i]);
		}
		search_time /= partition_number;

		for (int i = 0; i < partition_number; i++)
		{
			for (int j = 0; j < global_candidates.size(); j++)
			{
				for (int k = 0; k < candidates[i][j].size(); k++)
				{
					int path_ID = candidates[i][j][k];
					vector<int> tmp_path(partitions[i]->paths[path_ID], partitions[i]->paths[path_ID] + path_nodes);
					global_boundary[j].push_back(partitions[i]->path_boundary[path_ID]);

					for (int l = 0; l < tmp_path.size(); l++)
					{
						tmp_path[l] = partitions[i]->graph_extend_map[tmp_path[l]];
					}

					global_candidates[j].push_back(Candidate_Path(tmp_path));
				}
			}
		}

		double join_time = 0;
		for (int i = 0; i < global_candidates.size(); i++)
		{

			vector<Query_Path> query_plan;
			query_plan.push_back(query_graphs[query_ID]->query_plan[i]);
			for (int j = 0; j < query_graphs[query_ID]->query_plan.size(); j++)
			{
				if (j != i)
				{
					query_plan.push_back(query_graphs[query_ID]->query_plan[j]);
				}
			}

			vector<unordered_map<int, int>> key_cols(query_plan.size());
			for (int cur_path = 1; cur_path < query_plan.size(); cur_path++)
			{
				for (int cur_node = 0; cur_node < path_nodes; cur_node++)
				{
					for (int j = 0; j < cur_path; j++)
					{
						for (int k = 0; k < path_nodes; k++)
						{
							if (query_plan[cur_path].nodes[cur_node] == query_plan[j].nodes[k])
							{
								auto it = key_cols[cur_path].find(cur_node);
								if (it == key_cols[cur_path].end())
								{
									key_cols[cur_path].insert({cur_node, j * path_nodes + k});
								}
							}
						}
					}
				}
			}

			vector<vector<Candidate_Path>> global_candidates_resort;
			global_candidates_resort.push_back(global_candidates[i]);
			for (int j = 0; j < query_graphs[query_ID]->query_plan.size(); j++)
			{
				if (j != i)
				{
					global_candidates_resort.push_back(global_candidates[j]);
				}
			}

			vector<vector<vector<int>>> tables(global_candidates.size());
			for (int j = 0; j < global_candidates.size(); j++)
			{
				if (j == i)
				{
					for (int k = 0; k < global_candidates[j].size(); k++)
					{
						if (global_boundary[j][k] == 1)
						{
							tables[j].push_back(global_candidates[j][k].nodes);
						}
					}
				}
				else
				{
					for (int k = 0; k < global_candidates[j].size(); k++)
					{
						tables[j].push_back(global_candidates[j][k].nodes);
					}
				}
			}
			vector<vector<int>> result;

			unordered_map<Key, vector<int>> hash_map;

			vector<int> col_indices_a;
			vector<int> col_indices_b;

			typedef chrono::high_resolution_clock Clock;
			auto start_time = Clock::now();

			for (const auto &pair : key_cols[1])
			{
				col_indices_a.push_back(pair.second);
				col_indices_b.push_back(pair.first);
			}
			vector<vector<int>> keys_a(tables[0].size(), vector<int>(col_indices_a.size()));
			transform(tables[0].begin(), tables[0].end(), keys_a.begin(),
					  [&col_indices_a](const vector<int> &row)
					  {
						  vector<int> col_data;
						  for (int col_index : col_indices_a)
						  {
							  col_data.push_back(row.at(col_index));
						  }
						  return col_data;
					  });
			vector<vector<int>> keys_b(tables[1].size(), vector<int>(col_indices_b.size()));
			transform(tables[1].begin(), tables[1].end(), keys_b.begin(),
					  [&col_indices_b](const vector<int> &row)
					  {
						  vector<int> col_data;
						  for (int col_index : col_indices_b)
						  {
							  col_data.push_back(row.at(col_index));
						  }
						  return col_data;
					  });

			for (int i = 0; i < tables[1].size(); ++i)
			{
				Key key{keys_b[i]};
				hash_map[key].push_back(i);
			}

			for (int i = 0; i < tables[0].size(); ++i)
			{
				if (result.size() > table_max_size)
				{
					break;
				}

				Key key{keys_a[i]};
				if (hash_map.count(key))
				{
					for (int j : hash_map[key])
					{
						if (result.size() > table_max_size)
						{
							break;
						}

						vector<int> row;
						row.insert(row.end(), tables[0][i].begin(), tables[0][i].end());
						row.insert(row.end(), tables[1][j].begin(), tables[1][j].end());
						result.push_back(row);
					}
				}
			}

			for (int t = 2; t < tables.size(); ++t)
			{
				vector<vector<int>> tmp_result;
				hash_map.clear();
				col_indices_a.clear();
				col_indices_b.clear();

				for (const auto &pair : key_cols[t])
				{
					col_indices_a.push_back(pair.second);
					col_indices_b.push_back(pair.first);
				}

				vector<vector<int>> keys_a(result.size(), vector<int>(col_indices_a.size()));
				transform(result.begin(), result.end(), keys_a.begin(),
						  [&col_indices_a](const vector<int> &row)
						  {
							  vector<int> col_data;
							  for (int col_index : col_indices_a)
							  {
								  col_data.push_back(row.at(col_index));
							  }
							  return col_data;
						  });

				vector<vector<int>> keys_b(tables[t].size(), vector<int>(col_indices_b.size()));
				transform(tables[t].begin(), tables[t].end(), keys_b.begin(),
						  [&col_indices_b](const vector<int> &row)
						  {
							  vector<int> col_data;
							  for (int col_index : col_indices_b)
							  {
								  col_data.push_back(row.at(col_index));
							  }
							  return col_data;
						  });

				int p = result.size();
				for (int i = 0; i < p; ++i)
				{
					Key key{keys_a[i]};
					hash_map[key].push_back(i);
				}

				for (int i = 0; i < tables[t].size(); ++i)
				{
					if (tmp_result.size() > table_max_size)
					{
						break;
					}

					Key key{keys_b[i]};
					if (hash_map.count(key))
					{
						for (int j : hash_map[key])
						{
							if (tmp_result.size() > table_max_size)
							{
								break;
							}
							vector<int> row;
							row.insert(row.end(), result[j].begin(), result[j].end());
							row.insert(row.end(), tables[t][i].begin(), tables[t][i].end());
							tmp_result.push_back(row);
						}
					}
				}
				result = tmp_result;
			}
			auto end_time = Clock::now();
			join_time += chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count() / 1e+6;
		}
		join_time /= global_candidates.size();

		query_time += query_plan_time + search_time + join_time;

		cout << "query graph ID: " << query_ID << endl;

		cout << "query plane generation time: " << query_plan_time << " index search time: " << search_time << " hash join time: " << join_time << " query time: " << query_plan_time + search_time + join_time << endl;
	}

	cout << endl
		 << "Query Number: " << query_number << endl;
	printf("Average Query Time(ms): %.4f\n", query_time / (1.0 * query_number));

	return 0;
}