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
#include <algorithm>
#include <set>
#include <vector>
#include <chrono>

using namespace std;
using namespace chrono;

int table_max_size = 100000;

struct Key
{
	vector<int> values;

	bool operator==(const Key &other) const
	{
		return values == other.values;
	}
};

namespace std
{
	template <>
	struct hash<Key>
	{
		size_t operator()(const Key &key) const
		{
			string str;
			for (int value : key.values)
			{
				str += to_string(value) + "_";
			}
			return hash<string>()(str);
		}
	};
}

struct Query_Path
{
	vector<int> nodes;
	int weight;

	Query_Path(vector<int> nodes, int weight) : nodes(nodes), weight(weight) {}

	bool operator<(const Query_Path &other)
	{
		if (weight == other.weight)
		{
			for (int i = 0; i < nodes.size(); i++)
			{
				if (nodes[i] != other.nodes[i])
				{
					return nodes[i] < other.nodes[i];
				}
			}
		}
		return weight > other.weight;
	}
};

struct Candidate_Path
{
	vector<int> nodes;

	Candidate_Path() {}

	Candidate_Path(vector<int> nodes) : nodes(nodes) {}

	bool operator<(const Candidate_Path &other)
	{
		for (int i = 0; i < nodes.size(); i++)
		{
			if (nodes[i] == other.nodes[i])
			{
				return false;
			}
			else
			{
				return nodes[i] < other.nodes[i];
			}
		}
	}
};

struct CandidatePathComparator
{
	bool operator()(const Candidate_Path &a, const Candidate_Path &b) const
	{
		return a.nodes == b.nodes;
	}
};

class Query_Graph
{
public:
	Graph *graph;
	int *map;

	int num_paths, path_length;
	vector<Query_Path> paths;

	vector<Query_Path> query_plan;

	Query_Graph() {}

	Query_Graph(string path, int idx)
	{
		string graph_file_name = path + "query_graph-" + to_string(idx) + ".txt";
		string map_file_name = path + "query_graph_map-" + to_string(idx) + ".txt";
		string paths_file_name = path + "query_graph_paths-" + to_string(idx) + ".txt";
		graph = new Graph(true);
		graph->loadGraphFromFile(graph_file_name);

		ifstream ifs;
		ifs.open(map_file_name, ios::in);
		int num_nodes;
		ifs >> num_nodes;
		map = new int[num_nodes];
		for (int i = 0; i < num_nodes; i++)
		{
			int number;
			ifs >> number >> map[i];
		}
		ifs.close();

		ifs.open(paths_file_name, ios::in);
		ifs >> num_paths >> path_length;
		for (int i = 0; i < num_paths; i++)
		{
			vector<int> nodes(path_length);
			int weight;
			for (int j = 0; j < path_length; j++)
			{
				ifs >> nodes[j];
			}
			ifs >> weight;
			paths.push_back(Query_Path(nodes, weight));
		}
		sort(paths.begin(), paths.end());
	}

	~Query_Graph()
	{
		delete[] graph;
		delete[] map;
	}

	vector<int> overlap_nodes(vector<int> a, vector<int> b)
	{
		sort(a.begin(), a.end());
		sort(b.begin(), b.end());
		vector<int> c;
		set_intersection(a.begin(), a.end(), b.begin(), b.end(), back_inserter(c));
		return c;
	}

	void generate_query_plan()
	{
		int start = 0;
		for (int i = 0; i < graph->vertices_count_; i++)
		{
			if (graph->getVertexDegree(i) == graph->max_degree_)
			{
				start = i;
				break;
			}
		}
		vector<int> query_plan_nodes;
		bool flag = false;
		for (vector<Query_Path>::iterator it = paths.begin(); it != paths.end(); it++)
		{
			for (int j = 0; j < path_length; j++)
			{
				if (it->nodes[j] == start)
				{
					query_plan_nodes = it->nodes;
					query_plan.push_back(*it);
					paths.erase(it);
					flag = true;
					break;
				}
			}
			if (flag)
			{
				break;
			}
		}

		while (query_plan_nodes.size() != graph->vertices_count_)
		{
			flag = false;
			for (int overlap = 1; overlap < path_length; overlap++)
			{
				for (vector<Query_Path>::iterator it = paths.begin(); it != paths.end(); it++)
				{
					int overlap_num = overlap_nodes(query_plan_nodes, it->nodes).size();
					if (overlap_num == overlap)
					{
						query_plan.push_back(*it);
						query_plan_nodes.insert(query_plan_nodes.end(), it->nodes.begin(), it->nodes.end());
						set<int> temp(query_plan_nodes.begin(), query_plan_nodes.end());
						query_plan_nodes.assign(temp.begin(), temp.end());
						paths.erase(it);
						flag = true;
						break;
					}
				}
				if (flag)
				{
					break;
				}
			}
		}
	}

	double generate_query_plan_2()
	{
		int start = 0;

		typedef chrono::high_resolution_clock Clock;
		auto start_time = Clock::now();

		for (int i = 0; i < graph->vertices_count_; i++)
		{
			if (graph->getVertexDegree(i) == graph->max_degree_)
			{
				start = i;
				break;
			}
		}
		vector<Query_Path> initial_paths;
		for (int i = 0; i < paths.size(); i++)
		{
			for (int j = 0; j < path_length; j++)
			{
				if (paths[i].nodes[j] == start)
				{
					initial_paths.push_back(paths[i]);
					break;
				}
			}
		}
		int global_cost = -100000000;
		for (int i = 0; i < initial_paths.size(); i++)
		{
			vector<int> query_plan_nodes;
			vector<Query_Path> local_query_plan;
			local_query_plan.push_back(initial_paths[i]);
			for (int j = 0; j < path_length; j++)
			{
				query_plan_nodes.push_back(initial_paths[i].nodes[j]);
			}
			int local_cost = initial_paths[i].weight;
			vector<Query_Path> paths_beifen = paths;

			while (query_plan_nodes.size() != graph->vertices_count_)
			{
				bool flag = false;
				for (int overlap = 1; overlap < path_length; overlap++)
				{
					for (vector<Query_Path>::iterator it = paths_beifen.begin(); it != paths_beifen.end(); it++)
					{
						int overlap_num = overlap_nodes(query_plan_nodes, it->nodes).size();
						if (overlap_num == overlap)
						{
							local_query_plan.push_back(*it);
							query_plan_nodes.insert(query_plan_nodes.end(), it->nodes.begin(), it->nodes.end());
							set<int> temp(query_plan_nodes.begin(), query_plan_nodes.end());
							query_plan_nodes.assign(temp.begin(), temp.end());
							paths_beifen.erase(it);
							flag = true;
							local_cost += it->weight;
							break;
						}
					}
					if (flag)
					{
						break;
					}
				}
			}
			if (local_cost > global_cost)
			{
				global_cost = local_cost;
				query_plan = local_query_plan;
			}
		}

		auto end_time = Clock::now();
		return chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count() / 1e+6;
	}

	void generate_query_plan_3(int epsilon)
	{
		int start = 0;
		for (int i = 0; i < graph->vertices_count_; i++)
		{
			if (graph->getVertexDegree(i) == graph->max_degree_)
			{
				start = i;
				break;
			}
		}
		vector<Query_Path> initial_paths_complete;
		for (int i = 0; i < paths.size(); i++)
		{
			for (int j = 0; j < path_length; j++)
			{
				if (paths[i].nodes[j] == start)
				{
					initial_paths_complete.push_back(paths[i]);
					break;
				}
			}
		}
		srand(2022);
		vector<Query_Path> initial_paths;
		for (int i = 0; i < epsilon; i++)
		{
			initial_paths.push_back(initial_paths_complete[rand() % initial_paths_complete.size()]);
		}

		vector<Query_Path> local_query_plan;
		int global_cost = -100000000;
		for (int i = 0; i < initial_paths.size(); i++)
		{
			vector<int> query_plan_nodes;
			local_query_plan.push_back(initial_paths[i]);
			for (int j = 0; j < path_length; j++)
			{
				query_plan_nodes.push_back(initial_paths[i].nodes[j]);
			}
			int local_cost = initial_paths[i].weight;
			vector<Query_Path> paths_beifen = paths;

			while (query_plan_nodes.size() != graph->vertices_count_)
			{
				bool flag = false;
				for (int overlap = 1; overlap < path_length; overlap++)
				{
					for (vector<Query_Path>::iterator it = paths_beifen.begin(); it != paths_beifen.end(); it++)
					{
						int overlap_num = overlap_nodes(query_plan_nodes, it->nodes).size();
						if (overlap_num == overlap)
						{
							local_query_plan.push_back(*it);
							query_plan_nodes.insert(query_plan_nodes.end(), it->nodes.begin(), it->nodes.end());
							set<int> temp(query_plan_nodes.begin(), query_plan_nodes.end());
							query_plan_nodes.assign(temp.begin(), temp.end());
							paths_beifen.erase(it);
							flag = true;
							local_cost += it->weight;
							break;
						}
					}
					if (flag)
					{
						break;
					}
				}
			}
			if (local_cost > global_cost)
			{
				global_cost = local_cost;
				query_plan = local_query_plan;
			}
		}
	}
};

class Model
{
public:
	int embedding_num, embedding_dim;
	int query_num, query_size;
	double **one_embeddings;
	double **zero_embeddings;
	double ***query_one_embeddings;
	double ***query_zero_embeddings;

	Model() {}

	Model(string path, int query_num)
	{
		this->query_num = query_num;
		string one_embeddings_file = path + "one_embeddings.txt";
		string zero_embeddings_file = path + "zero_embeddings.txt";

		ifstream ifs;
		ifs.open(one_embeddings_file, ios::in);
		ifs >> embedding_num >> embedding_dim;
		one_embeddings = new double *[embedding_num];
		for (int i = 0; i < embedding_num; i++)
		{
			one_embeddings[i] = new double[embedding_dim];
			for (int j = 0; j < embedding_dim; j++)
			{
				ifs >> one_embeddings[i][j];
			}
		}
		ifs.close();

		ifs.open(zero_embeddings_file, ios::in);
		ifs >> embedding_num >> embedding_dim;
		zero_embeddings = new double *[embedding_num];
		for (int i = 0; i < embedding_num; i++)
		{
			zero_embeddings[i] = new double[embedding_dim];
			for (int j = 0; j < embedding_dim; j++)
			{
				ifs >> zero_embeddings[i][j];
			}
		}
		ifs.close();

		query_one_embeddings = new double **[query_num];
		query_zero_embeddings = new double **[query_num];
		string query_embeddings_path = path + "/query_embeddings/";
		for (int i = 0; i < query_num; i++)
		{
			string query_one_embedding_file = query_embeddings_path + "query_embeddings-" + to_string(i) + ".txt";
			ifs.open(query_one_embedding_file);
			ifs >> query_size >> embedding_dim;
			query_one_embeddings[i] = new double *[query_size];
			for (int j = 0; j < query_size; j++)
			{
				query_one_embeddings[i][j] = new double[embedding_dim];
				for (int k = 0; k < embedding_dim; k++)
				{
					ifs >> query_one_embeddings[i][j][k];
				}
			}
			ifs.close();

			string query_zero_embedding_file = query_embeddings_path + "query_zero_embeddings-" + to_string(i) + ".txt";
			ifs.open(query_zero_embedding_file);
			ifs >> query_size >> embedding_dim;
			query_zero_embeddings[i] = new double *[query_size];
			for (int j = 0; j < query_size; j++)
			{
				query_zero_embeddings[i][j] = new double[embedding_dim];
				for (int k = 0; k < embedding_dim; k++)
				{
					ifs >> query_zero_embeddings[i][j][k];
				}
			}
			ifs.close();
		}
	}

	~Model()
	{
		for (int i = 0; i < embedding_num; i++)
		{
			delete[] one_embeddings[i];
			delete[] zero_embeddings[i];
		}
		delete[] one_embeddings;
		delete[] zero_embeddings;
		for (int i = 0; i < query_num; i++)
		{
			for (int j = 0; j < query_size; j++)
			{
				delete[] query_one_embeddings[i][j];
				delete[] query_zero_embeddings[i][j];
			}
		}
		for (int i = 0; i < query_num; i++)
		{
			delete[] query_one_embeddings[i];
			delete[] query_zero_embeddings[i];
		}
		delete[] query_one_embeddings;
		delete[] query_zero_embeddings;
	}
};

class Auxiliary_Data
{
public:
	int main_dim, associate_dim;
	double *mbr_zero_main;
	double *mbr_zero_associate;
	double *mbr_associate;

	int *degrees;

	Auxiliary_Data() {}

	Auxiliary_Data(int main_dim, int associate_dim, int path_length)
	{
		this->main_dim = main_dim;
		this->associate_dim = associate_dim;

		mbr_zero_main = new double[main_dim * 2];
		mbr_associate = new double[associate_dim * 2];
		mbr_zero_associate = new double[associate_dim * 2];

		degrees = new int[path_length];
	}

	~Auxiliary_Data()
	{
		delete[] mbr_zero_main;
		delete[] mbr_zero_associate;
		delete[] mbr_associate;
		delete[] degrees;
	}
};

class Query_Path_MBR
{
public:
	int main_dim, associate_dim, ID, path_length;
	double key;

	vector<double> mbr_main;
	vector<double> mbr_associate;
	vector<double> mbr_zero_main;
	vector<double> mbr_zero_associate;
	vector<int> node_degrees;

	Query_Path_MBR() {}

	Query_Path_MBR(int ID, int main_dim, int associate_dim, int path_length)
	{
		this->ID = ID;
		this->main_dim = main_dim;
		this->associate_dim = associate_dim;
		this->path_length = path_length;
	}

	void calculate_key()
	{
		key = 0;
		for (int i = 0; i < main_dim; i++)
		{
			key += -abs(mbr_main[i * 2]);
		}
	}
};

class Query_Plan
{
public:
	vector<Query_Path_MBR> query_mbrs;
	double key;

	Query_Plan() {}

	Query_Plan(vector<Query_Path_MBR> mbrs)
	{
		query_mbrs = mbrs;
	}

	void calculate_key()
	{
		key = -10000;
		for (int i = 0; i < query_mbrs.size(); i++)
		{
			if (query_mbrs[i].key > key)
			{
				key = query_mbrs[i].key;
			}
		}
	}

	void calculate_key(int length)
	{
		key = -10000;
		for (int i = 0; i < length; i++)
		{
			if (query_mbrs[i].key > key)
			{
				key = query_mbrs[i].key;
			}
		}
	}
};

class Partition
{
public:
	string file_path;
	Graph *graph;
	Graph *graph_extend;
	int *graph_map;
	int *graph_extend_map;

	int path_num, path_length;
	int **paths;

	int model_num;
	Model **models;

	RTree *rtree_index;
	int path_main_dim, path_associate_dim;

	double **path_one_main_embeddings;
	double **path_one_associate_embeddings;
	double **path_zero_main_embeddings;
	double **path_zero_associate_embeddings;

	double *path_one_main_bounces;
	double *path_one_associate_bounces;
	double *path_zero_main_bounces;
	double *path_zero_associate_bounces;

	int **path_degrees;

	int embedding_dim;

	int nodes_num;
	Auxiliary_Data **auxiliary_index;

	int max_node_ID, min_node_ID;

	double *node_keys;

	int *path_boundary;

	Partition() {}

	Partition(string file_path, int model_num, int query_num, Graph *data_graph)
	{
		this->file_path = file_path;
		this->model_num = model_num;

		graph = new Graph(true);
		string filename = file_path + "graph.txt";
		graph->loadGraphFromFile(filename);
		graph_extend = new Graph(true);
		filename = file_path + "graph_extend.txt";
		graph_extend->loadGraphFromFile(filename);

		filename = file_path + "graph_map.txt";
		ifstream ifs;
		ifs.open(filename, ios::in);
		int num_nodes;
		ifs >> num_nodes;
		graph_map = new int[num_nodes];
		for (int i = 0; i < num_nodes; i++)
		{
			int number;
			ifs >> number >> graph_map[i];
		}
		ifs.close();
		filename = file_path + "graph_extend_map.txt";
		ifs.open(filename, ios::in);
		ifs >> num_nodes;
		graph_extend_map = new int[num_nodes];
		for (int i = 0; i < num_nodes; i++)
		{
			int number;
			ifs >> number >> graph_extend_map[i];
		}
		ifs.close();

		filename = file_path + "paths.txt";
		ifs.open(filename, ios::in);
		ifs >> path_num >> path_length;

		paths = new int *[path_num];
		path_degrees = new int *[path_num];
		for (int i = 0; i < path_num; i++)
		{
			paths[i] = new int[path_length];
			path_degrees[i] = new int[path_length];
			for (int j = 0; j < path_length; j++)
			{
				ifs >> paths[i][j];
				path_degrees[i][j] = data_graph->getVertexDegree(graph_extend_map[paths[i][j]]);
			}
		}

		models = new Model *[model_num];
		for (int i = 0; i < model_num; i++)
		{
			string model_path = file_path + "Model-" + to_string(i) + "/";
			models[i] = new Model(model_path, query_num);
		}
		embedding_dim = models[0]->embedding_dim;

		path_main_dim = embedding_dim * path_length;
		path_associate_dim = embedding_dim * path_length * (model_num - 1);

		path_one_main_embeddings = new double *[path_num];
		path_one_associate_embeddings = new double *[path_num];
		path_zero_main_embeddings = new double *[path_num];
		path_zero_associate_embeddings = new double *[path_num];

		for (int i = 0; i < path_num; i++)
		{
			path_one_main_embeddings[i] = new double[path_main_dim];
			path_one_associate_embeddings[i] = new double[path_associate_dim];
			path_zero_main_embeddings[i] = new double[path_main_dim];
			path_zero_associate_embeddings[i] = new double[path_associate_dim];

			for (int j = 0; j < path_length; j++)
			{
				int node = paths[i][j];
				for (int k = 0; k < embedding_dim; k++)
				{
					path_one_main_embeddings[i][j * embedding_dim + k] = models[0]->one_embeddings[node][k];
					path_zero_main_embeddings[i][j * embedding_dim + k] = models[0]->zero_embeddings[node][k];
				}
				for (int k = 0; k + 1 < model_num; k++)
				{
					for (int l = 0; l < embedding_dim; l++)
					{
						path_one_associate_embeddings[i][(j * (model_num - 1) + k) * embedding_dim + l] = models[k + 1]->one_embeddings[node][l];
						path_zero_associate_embeddings[i][(j * (model_num - 1) + k) * embedding_dim + l] = models[k + 1]->zero_embeddings[node][l];
					}
				}
			}
		}

		path_one_main_bounces = new double[path_main_dim];
		path_one_associate_bounces = new double[path_associate_dim];
		path_zero_main_bounces = new double[path_main_dim];
		path_zero_associate_bounces = new double[path_associate_dim];

		for (int i = 0; i < path_main_dim; i++)
		{
			path_one_main_bounces[i] = 0;
			path_zero_main_bounces[i] = 0;
		}
		for (int i = 0; i < path_associate_dim; i++)
		{
			path_one_associate_bounces[i] = 0;
			path_zero_associate_bounces[i] = 0;
		}

		for (int i = 0; i < path_num; i++)
		{
			for (int j = 0; j < path_main_dim; j++)
			{
				if (path_one_main_embeddings[i][j] > path_one_main_bounces[j])
				{
					path_one_main_bounces[j] = path_one_main_embeddings[i][j];
				}
				if (path_zero_main_embeddings[i][j] > path_zero_main_bounces[j])
				{
					path_zero_main_bounces[j] = path_zero_main_embeddings[i][j];
				}
			}
			for (int j = 0; j < path_associate_dim; j++)
			{
				if (path_one_associate_embeddings[i][j] > path_one_associate_bounces[j])
				{
					path_one_associate_bounces[j] = path_one_associate_embeddings[i][j];
				}
				if (path_zero_associate_embeddings[i][j] > path_zero_associate_bounces[j])
				{
					path_zero_associate_bounces[j] = path_zero_associate_embeddings[i][j];
				}
			}
		}

		path_boundary = new int[path_num];
		for (int i = 0; i < path_num; i++)
		{
			bool flag = false;
			for (int j = 0; j < path_length; j++)
			{
				int node_ID = graph_extend_map[paths[i][j]];
				int k = 0;
				for (; k < graph->vertices_count_; k++)
				{
					if (node_ID == graph_map[k])
					{
						break;
					}
				}
				if (k == graph->vertices_count_)
				{
					flag = true;
					break;
				}
			}
			if (flag)
			{
				path_boundary[i] = 1;
			}
			else
			{
				path_boundary[i] = 0;
			}
		}
	}

	~Partition()
	{
		delete graph;
		delete graph_extend;
		delete graph_map;
		delete graph_extend_map;

		for (int i = 0; i < path_num; i++)
		{
			delete[] paths[i];
		}
		delete[] paths;

		for (int i = 0; i < model_num; i++)
		{
			delete models[i];
		}
		delete[] models;

		for (int i = 0; i < path_num; i++)
		{
			delete[] path_one_main_embeddings[i];
			delete[] path_one_associate_embeddings[i];
			delete[] path_zero_main_embeddings[i];
			delete[] path_zero_associate_embeddings[i];
		}
		delete[] path_one_main_embeddings;
		delete[] path_one_associate_embeddings;
		delete[] path_zero_main_embeddings;
		delete[] path_zero_associate_embeddings;

		delete[] path_one_main_bounces;
		delete[] path_one_associate_bounces;
		delete[] path_zero_main_bounces;
		delete[] path_zero_associate_bounces;

		delete[] path_boundary;
	}

	void build_auxiliary_index(RTNode *rtn, int current_ID)
	{
		if (rtn->level == 0)
		{
			for (int i = 0; i < rtn->num_entries; i++)
			{
				int data_ID = rtn->entries[i].son;

				for (int j = 0; j < path_length; j++)
				{
					if (i == 0)
					{
						auxiliary_index[current_ID]->degrees[j] = path_degrees[data_ID][j];
					}
					else
					{
						if (auxiliary_index[current_ID]->degrees[j] < path_degrees[data_ID][j])
						{
							auxiliary_index[current_ID]->degrees[j] = path_degrees[data_ID][j];
						}
					}
				}

				for (int j = 0; j < path_associate_dim; j++)
				{
					if (i == 0)
					{
						auxiliary_index[current_ID]->mbr_associate[2 * j] = path_one_associate_embeddings[data_ID][j];
						auxiliary_index[current_ID]->mbr_associate[2 * j + 1] = path_one_associate_embeddings[data_ID][j];
						auxiliary_index[current_ID]->mbr_zero_associate[2 * j] = path_zero_associate_embeddings[data_ID][j];
						auxiliary_index[current_ID]->mbr_zero_associate[2 * j + 1] = path_zero_associate_embeddings[data_ID][j];
					}
					else
					{
						if (auxiliary_index[current_ID]->mbr_associate[2 * j] > path_one_associate_embeddings[data_ID][j])
						{
							auxiliary_index[current_ID]->mbr_associate[2 * j] = path_one_associate_embeddings[data_ID][j];
						}
						if (auxiliary_index[current_ID]->mbr_associate[2 * j + 1] < path_one_associate_embeddings[data_ID][j])
						{
							auxiliary_index[current_ID]->mbr_associate[2 * j + 1] = path_one_associate_embeddings[data_ID][j];
						}
						if (auxiliary_index[current_ID]->mbr_zero_associate[2 * j] > path_zero_associate_embeddings[data_ID][j])
						{
							auxiliary_index[current_ID]->mbr_zero_associate[2 * j] = path_zero_associate_embeddings[data_ID][j];
						}
						if (auxiliary_index[current_ID]->mbr_zero_associate[2 * j + 1] < path_zero_associate_embeddings[data_ID][j])
						{
							auxiliary_index[current_ID]->mbr_zero_associate[2 * j + 1] = path_zero_associate_embeddings[data_ID][j];
						}
					}
				}
				for (int j = 0; j < path_main_dim; j++)
				{
					if (i == 0)
					{
						auxiliary_index[current_ID]->mbr_zero_main[2 * j] = path_zero_main_embeddings[data_ID][j];
						auxiliary_index[current_ID]->mbr_zero_main[2 * j + 1] = path_zero_main_embeddings[data_ID][j];
					}
					else
					{
						if (auxiliary_index[current_ID]->mbr_zero_main[2 * j] > path_zero_main_embeddings[data_ID][j])
						{
							auxiliary_index[current_ID]->mbr_zero_main[2 * j] = path_zero_main_embeddings[data_ID][j];
						}
						if (auxiliary_index[current_ID]->mbr_zero_main[2 * j + 1] < path_zero_main_embeddings[data_ID][j])
						{
							auxiliary_index[current_ID]->mbr_zero_main[2 * j + 1] = path_zero_main_embeddings[data_ID][j];
						}
					}
				}
			}
		}
		else
		{
			for (int i = 0; i < rtn->num_entries; i++)
			{
				RTNode *child = new RTNode(rtn->my_tree, rtn->entries[i].son);
				build_auxiliary_index(child, rtn->entries[i].son);

				node_keys[rtn->entries[i].son] = -L1_norm(rtn->entries[i].bounces, rtree_index->dimension);

				int node_ID = rtn->entries[i].son;

				for (int j = 0; j < path_length; j++)
				{
					if (i == 0)
					{
						auxiliary_index[current_ID]->degrees[j] = auxiliary_index[node_ID]->degrees[j];
					}
					else
					{
						if (auxiliary_index[current_ID]->degrees[j] < auxiliary_index[node_ID]->degrees[j])
						{
							auxiliary_index[current_ID]->degrees[j] = auxiliary_index[node_ID]->degrees[j];
						}
					}
				}

				for (int j = 0; j < path_associate_dim; j++)
				{
					if (i == 0)
					{
						auxiliary_index[current_ID]->mbr_associate[2 * j] = auxiliary_index[node_ID]->mbr_associate[2 * j];
						auxiliary_index[current_ID]->mbr_associate[2 * j + 1] = auxiliary_index[node_ID]->mbr_associate[2 * j + 1];
						auxiliary_index[current_ID]->mbr_zero_associate[2 * j] = auxiliary_index[node_ID]->mbr_zero_associate[2 * j];
						auxiliary_index[current_ID]->mbr_zero_associate[2 * j + 1] = auxiliary_index[node_ID]->mbr_zero_associate[2 * j + 1];
					}
					else
					{
						if (auxiliary_index[current_ID]->mbr_associate[2 * j] > auxiliary_index[node_ID]->mbr_associate[2 * j])
						{
							auxiliary_index[current_ID]->mbr_associate[2 * j] = auxiliary_index[node_ID]->mbr_associate[2 * j];
						}
						if (auxiliary_index[current_ID]->mbr_associate[2 * j + 1] < auxiliary_index[node_ID]->mbr_associate[2 * j + 1])
						{
							auxiliary_index[current_ID]->mbr_associate[2 * j + 1] = auxiliary_index[node_ID]->mbr_associate[2 * j + 1];
						}
						if (auxiliary_index[current_ID]->mbr_zero_associate[2 * j] > auxiliary_index[node_ID]->mbr_zero_associate[2 * j])
						{
							auxiliary_index[current_ID]->mbr_zero_associate[2 * j] = auxiliary_index[node_ID]->mbr_zero_associate[2 * j];
						}
						if (auxiliary_index[current_ID]->mbr_zero_associate[2 * j + 1] < auxiliary_index[node_ID]->mbr_zero_associate[2 * j + 1])
						{
							auxiliary_index[current_ID]->mbr_zero_associate[2 * j + 1] = auxiliary_index[node_ID]->mbr_zero_associate[2 * j + 1];
						}
					}
				}
				for (int j = 0; j < path_main_dim; j++)
				{
					if (i == 0)
					{
						auxiliary_index[current_ID]->mbr_zero_main[2 * j] = auxiliary_index[node_ID]->mbr_zero_main[2 * j];
						auxiliary_index[current_ID]->mbr_zero_main[2 * j + 1] = auxiliary_index[node_ID]->mbr_zero_main[2 * j + 1];
					}
					else
					{
						if (auxiliary_index[current_ID]->mbr_zero_main[2 * j] > auxiliary_index[node_ID]->mbr_zero_main[2 * j])
						{
							auxiliary_index[current_ID]->mbr_zero_main[2 * j] = auxiliary_index[node_ID]->mbr_zero_main[2 * j];
						}
						if (auxiliary_index[current_ID]->mbr_zero_main[2 * j + 1] < auxiliary_index[node_ID]->mbr_zero_main[2 * j + 1])
						{
							auxiliary_index[current_ID]->mbr_zero_main[2 * j + 1] = auxiliary_index[node_ID]->mbr_zero_main[2 * j + 1];
						}
					}
				}

				rtn->entries[i].del_son();
				delete child;
			}
		}
	}

	void build_index()
	{
		string file_path = this->file_path + "index.dat";
		char *index_file_name = new char[file_path.length() + 1];
		strcpy(index_file_name, file_path.c_str());

		fstream in(index_file_name, ios_base::in);
		bool generate_tree = true;
		if (!in)
		{

			in.close();
			generate_tree = true;
		}
		else
		{
			generate_tree = false;
			in.close();
		}

		if (generate_tree)
		{
			Entry *node;
			double index_time = 0;
			rtree_index = new RTree(index_file_name, 8192, NULL, path_main_dim);
			for (int i = 0; i < path_num; i++)
			{
				node = new Entry(path_main_dim, NULL);
				node->son = i;
				for (int j = 0; j < path_main_dim; j++)
				{
					node->bounces[2 * j] = path_one_main_embeddings[i][j];
					node->bounces[2 * j + 1] = path_one_main_embeddings[i][j];
				}
				double ind_st = clock();

				rtree_index->insert(node);

				double ind_ed = clock();
				index_time += (ind_ed - ind_st) / CLOCKS_PER_SEC;
			}
			cout << "index construction complete" << endl;
			cout << "index construction time: " << index_time << endl;
			delete rtree_index;
		}

		rtree_index = new RTree(index_file_name, NULL);
		cout << "index loading complete" << endl;

		rtree_index->load_root();
		nodes_num = 0;
		nodes_num = rtree_index->root_ptr->get_num_of_node();
		cout << "节点数量：" << nodes_num << endl;
		cout << "数据数量：" << rtree_index->root_ptr->get_num_of_data() << endl;
		auxiliary_index = new Auxiliary_Data *[nodes_num];
		node_keys = new double[nodes_num];
		for (int i = 0; i < nodes_num; i++)
		{
			auxiliary_index[i] = new Auxiliary_Data(path_main_dim, path_associate_dim, path_length);
		}
		build_auxiliary_index(rtree_index->root_ptr, rtree_index->root);
		cout << "Auxiliary Index Construction Complete!" << endl
			 << endl;

		delete[] index_file_name;
	}

	double L1_norm(const double *emb, int dim)
	{
		double score = 0;
		for (int i = 0; i < dim; i++)
		{
			score += abs(emb[i * 2 + 1]);
		}
		return score;
	}

	double query(Query_Graph *query_graph, int query_graph_ID, vector<vector<int>> &candidates)
	{
		Query_Plan Q;
		for (int i = 0; i < query_graph->query_plan.size(); i++)
		{

			double *one_main = new double[path_main_dim];
			double *one_associate = new double[path_associate_dim];
			double *zero_main = new double[path_main_dim];
			double *zero_associate = new double[path_associate_dim];

			for (int j = 0; j < path_length; j++)
			{
				int node = query_graph->query_plan[i].nodes[j];
				for (int k = 0; k < embedding_dim; k++)
				{
					one_main[j * embedding_dim + k] = models[0]->query_one_embeddings[query_graph_ID][node][k];
					zero_main[j * embedding_dim + k] = models[0]->query_zero_embeddings[query_graph_ID][node][k];
				}
				for (int k = 0; k + 1 < model_num; k++)
				{
					for (int l = 0; l < embedding_dim; l++)
					{
						one_associate[(j * (model_num - 1) + k) * embedding_dim + l] = models[k + 1]->query_one_embeddings[query_graph_ID][node][l];
						zero_associate[(j * (model_num - 1) + k) * embedding_dim + l] = models[k + 1]->query_zero_embeddings[query_graph_ID][node][l];
					}
				}
			}

			Query_Path_MBR p_q(i, path_main_dim, path_associate_dim, path_length);
			for (int j = 0; j < path_main_dim; j++)
			{
				p_q.mbr_main.push_back(one_main[j]);
				p_q.mbr_main.push_back(path_one_main_bounces[j]);
				p_q.mbr_zero_main.push_back(zero_main[j]);
				p_q.mbr_zero_main.push_back(path_zero_main_bounces[j]);
			}
			for (int j = 0; j < path_associate_dim; j++)
			{
				p_q.mbr_associate.push_back(one_associate[j]);
				p_q.mbr_associate.push_back(path_one_associate_bounces[j]);
				p_q.mbr_zero_associate.push_back(zero_associate[j]);
				p_q.mbr_zero_associate.push_back(path_zero_associate_bounces[j]);
			}
			for (int j = 0; j < path_length; j++)
			{
				int node = query_graph->query_plan[i].nodes[j];
				p_q.node_degrees.push_back(query_graph->graph->getVertexDegree(node));
			}
			p_q.calculate_key();
			Q.query_mbrs.push_back(p_q);

			delete[] one_main;
			delete[] one_associate;
			delete[] zero_main;
			delete[] zero_associate;
		}
		Q.calculate_key();

		Heap *hp = new Heap();
		hp->init(rtree_index->dimension);

		HeapEntry *he = new HeapEntry();
		rtree_index->load_root();
		he->son1 = rtree_index->root;
		he->key = -1e20;
		he->level = 1;
		hp->insert(he);
		delete he;

		vector<Query_Plan> Q_map(nodes_num);
		Q_map[rtree_index->root] = Q;
		candidates.resize(query_graph->query_plan.size());

		RTNode *child;

		typedef std::chrono::high_resolution_clock Clock;
		auto start_time = Clock::now();

		int traversal_index_nodes = 0;

		while (hp->used > 0)
		{
			he = new HeapEntry();
			hp->remove(he);
			int son = he->son1;
			int level = he->level;
			double key = he->key;
			delete he;

			if (Q_map[son].key < key)
			{
				break;
			}

			if (level == 0)
			{
				child = new RTNode(rtree_index, son);
				for (int i = 0; i < child->num_entries; i++)
				{
					int path_ID = child->entries[i].son;
					for (int j = 0; j < Q_map[son].query_mbrs.size(); j++)
					{
						int k = 0;
						for (; k < path_main_dim; k++)
						{
							if (Q_map[son].query_mbrs[j].mbr_zero_main[2 * k] != path_zero_main_embeddings[path_ID][k])
							{
								break;
							}
						}
						if (k == path_main_dim)
						{
							k = 0;
							for (; k < path_main_dim; k++)
							{
								if (Q_map[son].query_mbrs[j].mbr_main[2 * k] > child->entries[i].bounces[2 * k + 1])
								{
									break;
								}
							}
							if (k == path_main_dim)
							{
								k = 0;
								for (; k < path_associate_dim; k++)
								{
									if (Q_map[son].query_mbrs[j].mbr_associate[2 * k] > path_one_associate_embeddings[path_ID][k])
									{
										break;
									}
								}
								if (k == path_associate_dim)
								{
									candidates[Q_map[son].query_mbrs[j].ID].push_back(path_ID);
								}
							}
						}
					}
				}
				delete child;
			}
			else
			{
				child = new RTNode(rtree_index, son);
				for (int i = 0; i < child->num_entries; i++)
				{
					int node_ID = child->entries[i].son;
					for (int j = 0; j < Q_map[son].query_mbrs.size(); j++)
					{
						int k = 0;
						for (; k < path_main_dim; k++)
						{
							if (Q_map[son].query_mbrs[j].mbr_zero_main[2 * k] > auxiliary_index[node_ID]->mbr_zero_main[2 * k + 1] || Q_map[son].query_mbrs[j].mbr_zero_main[2 * k] < auxiliary_index[node_ID]->mbr_zero_main[2 * k])
							{
								break;
							}
						}
						if (k == path_main_dim)
						{
							k = 0;
							for (; k < path_main_dim; k++)
							{
								if (Q_map[son].query_mbrs[j].mbr_main[2 * k] > child->entries[i].bounces[2 * k + 1])
								{
									break;
								}
							}
							if (k == path_main_dim)
							{
								k = 0;
								for (; k < path_associate_dim; k++)
								{
									if (Q_map[son].query_mbrs[j].mbr_associate[2 * k] > auxiliary_index[node_ID]->mbr_associate[2 * k + 1])
									{
										break;
									}
								}
								if (k == path_associate_dim)
								{
									Q_map[node_ID].query_mbrs.push_back(Q_map[son].query_mbrs[j]);
								}
							}
						}
					}
					if (Q_map[node_ID].query_mbrs.size() != 0)
					{
						Q_map[node_ID].calculate_key();
						he = new HeapEntry();
						he->son1 = child->entries[i].son;
						he->level = child->level - 1;
						he->key = node_keys[he->son1];
						hp->insert(he);
						delete he;
						traversal_index_nodes++;
					}
				}
				delete child;
			}
		}
		auto end_time = Clock::now();
		double search_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / 1e+6;
		delete hp;

		vector<vector<vector<int>>> tables(candidates.size());
		for (int i = 0; i < candidates.size(); i++)
		{
			for (int j = 0; j < candidates[i].size(); j++)
			{
				if (path_boundary[candidates[i][j]] == 0)
				{
					vector<int> tmp_path;
					for (int k = 0; k < path_length; k++)
					{
						tmp_path.push_back(paths[candidates[i][j]][k]);
					}
					tables[i].push_back(tmp_path);
				}
			}
		}

		vector<unordered_map<int, int>> key_cols(query_graph->query_plan.size());
		for (int cur_path = 1; cur_path < query_graph->query_plan.size(); cur_path++)
		{
			for (int cur_node = 0; cur_node < path_length; cur_node++)
			{
				for (int j = 0; j < cur_path; j++)
				{
					for (int k = 0; k < path_length; k++)
					{
						if (query_graph->query_plan[cur_path].nodes[cur_node] == query_graph->query_plan[j].nodes[k])
						{
							auto it = key_cols[cur_path].find(cur_node);
							if (it == key_cols[cur_path].end())
							{
								key_cols[cur_path].insert({cur_node, j * path_length + k});
							}
						}
					}
				}
			}
		}

		vector<vector<int>> result;

		unordered_map<Key, vector<int>> hash_map;

		vector<int> col_indices_a;
		vector<int> col_indices_b;

		typedef chrono::high_resolution_clock Clock;
		start_time = Clock::now();

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
		end_time = Clock::now();
		search_time += chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count() / 1e+6;

		return search_time;
	}
};
