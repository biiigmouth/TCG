/**
 * Framework for Threes! and its variants (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include "board.h"
#include "action.h"
#include "weight.h"

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args), alpha(0) {
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		
		std::string res = info; // comma-separated sizes, e.g., "65536,65536"
		for (char& ch : res)
			if (!std::isdigit(ch)) ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size));
		
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	float alpha;
};

//tuple network player
class tuple_player:public agent{

public:
	tuple_player(const std::string& args = "") : agent(args), alpha(0.1f/64.0f) {
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~tuple_player() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

	struct state{
		board after;
		int reward;
	};

	virtual action take_action(const board& before){
		//std::shuffle(opcode.begin(), opcode.end(), engine);
		int best_reward = -1;
		float best_value=-9999999;
		board after;
		int best_op = -1;
		for (int f_op : {0,1,2,3}) {
			board temp=before;
			int f_reward = temp.slide(f_op);
			if(f_reward==-1){
				continue;
			}
			float value=evaluate_score(temp);

			if(f_reward+value>best_reward+best_value){
				best_reward=f_reward;
				best_op=f_op;
				best_value=value;
				after=temp;
			}
		}
		if(best_op!=-1){
			struct state epi={after,best_reward};
			episode.push_back(epi);
			
		}
		
		return action::slide(best_op);
		
	}


	int evaluate_feature(board& after, int net_index[6]){
		return after(net_index[0])*16*16*16*16*16+after(net_index[1])*16*16*16*16+after(net_index[2])*16*16*16+
		after(net_index[3])*16*16+after(net_index[4])*16+after(net_index[5]);
	}

	float evaluate_score(board& after){
		float score=0;
		for(int i=0;i<64;i++){
			int j=i/8;
			score+=net[j][evaluate_feature(after,network_index[i])];
		}
		
		return score;
	}
	
	void train_weights(board& after, float target){
		float temp=evaluate_score(after);
		float err=target-temp;
		float adjust_value=err*alpha;
		for(int i=0;i<64;i++){
			int j=i/8;
			net[j][evaluate_feature(after,network_index[i])]+=adjust_value;
		}
		
		/*
		net[0][evaluate_feature(after,0,1,2,3,4,5)]+=adjust_value;
		net[0][evaluate_feature(after,4,5,6,7,8,9)]+=adjust_value;
		net[0][evaluate_feature(after,5,6,7,9,10,11)]+=adjust_value;
		net[0][evaluate_feature(after,9,10,11,13,14,15)]+=adjust_value;
		net[1][evaluate_feature(after,2,3,6,7,11,15)]+=adjust_value;
		net[1][evaluate_feature(after,1,2,5,6,10,14)]+=adjust_value;
		net[1][evaluate_feature(after,5,6,9,10,13,14)]+=adjust_value;
		net[1][evaluate_feature(after,4,5,8,9,12,13)]+=adjust_value;
		net[2][evaluate_feature(after,11,15,10,14,13,12)]+=adjust_value;
		net[2][evaluate_feature(after,7,11,6,10,9,8)]+=adjust_value;
		net[2][evaluate_feature(after,4,5,6,8,9,10)]+=adjust_value;
		net[2][evaluate_feature(after,0,1,2,4,5,6)]+=adjust_value;
		net[3][evaluate_feature(after,12,13,8,9,4,0)]+=adjust_value;
		net[3][evaluate_feature(after,13,14,9,10,5,1)]+=adjust_value;
		net[3][evaluate_feature(after,9,10,5,6,1,2)]+=adjust_value;
		net[3][evaluate_feature(after,10,11,6,7,2,3)]+=adjust_value;
		net[4][evaluate_feature(after,12,13,14,15,8,9)]+=adjust_value;
		net[4][evaluate_feature(after,8,9,10,11,4,5)]+=adjust_value;
		net[4][evaluate_feature(after,9,10,11,5,6,7)]+=adjust_value;
		net[4][evaluate_feature(after,5,6,7,1,2,3)]+=adjust_value;
		net[5][evaluate_feature(after,0,4,8,12,1,5)]+=adjust_value;
		net[5][evaluate_feature(after,1,5,9,13,2,6)]+=adjust_value;
		net[5][evaluate_feature(after,5,6,9,10,13,14)]+=adjust_value;
		net[5][evaluate_feature(after,6,7,10,11,14,15)]+=adjust_value;
		net[6][evaluate_feature(after,0,1,2,3,6,7)]+=adjust_value;
		net[6][evaluate_feature(after,4,5,6,7,10,11)]+=adjust_value;
		net[6][evaluate_feature(after,4,5,6,8,9,10)]+=adjust_value;
		net[6][evaluate_feature(after,8,9,10,12,13,14)]+=adjust_value;
		net[7][evaluate_feature(after,15,11,7,3,14,10)]+=adjust_value;
		net[7][evaluate_feature(after,14,10,6,2,13,9)]+=adjust_value;
		net[7][evaluate_feature(after,13,14,9,10,5,6)]+=adjust_value;
		net[7][evaluate_feature(after,12,13,8,9,4,5)]+=adjust_value;
		*/
	}

	virtual void open_episode(const std::string& flag = "") {
		episode.clear();
	}
	virtual void close_episode(const std::string& flag = "") {
		if(episode.empty()){
			return;
		}
		train_weights(episode[episode.size()-1].after,0);
		for(int i=episode.size()-2;i>=0;i--){
			train_weights(episode[i].after,episode[i+1].reward+evaluate_score(episode[i+1].after));
		}
	}

protected:
	virtual void init_weights(const std::string& info) {
		
		std::string res = info; // comma-separated sizes, e.g., "65536,65536"

		//using 16*16*16*16*16*16=16777216 for 8 networks
		for (char& ch : res)
			if (!std::isdigit(ch)) ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size));
		

	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	float alpha;
		//0 1 2 3
		//4 5 6 7
		//8 9 10 11
		//12 13 14 15
	std::vector<state> episode;
	int network_index[64][6]={
		{0,1,2,4,5,6},
		{2,3,6,7,10,11},
		{9,10,11,13,14,15},
		{4,5,8,9,12,13},
		{8,9,10,12,13,14},
		{0,1,4,5,8,9},
		{1,2,3,5,6,7},
		{6,7,10,11,14,15},

		{1,2,5,6,9,13},
		{4,5,6,7,10,11},
		{2,6,10,14,13,9},
		{4,5,8,9,10,11},
		{1,2,5,6,10,14},
		{6,7,8,9,10,11},
		{1,5,9,10,13,14},
		{4,5,6,7,8,9},

		{0,1,2,3,4,5},
		{2,6,3,7,11,15},
		{12,13,14,15,10,11},
		{0,4,8,12,9,13},
		{8,9,12,13,14,15},
		{0,1,4,5,8,12},
		{0,1,2,3,6,7},
		{3,7,10,11,14,15},

		{0,1,6,7,8,11},
		{3,7,6,9,10,14,},
		{5,8,9,10,15},
		{1,5,6,8,9,12},
		{6,9,10,11,12,13},
		{0,4,5,9,10,13},
		{2,3,4,5,6,9},
		{2,5,6,10,11,15},

		{0,1,2,5,9,10},
		{3,5,6,7,9,11},
		{5,6,10,13,14,15},
		{4,6,8,9,10,12},
		{5,6,9,12,13,14},
		{0,4,5,6,8,10},
		{1,2,3,6,9,10},
		{5,7,9,10,11,15},

		{0,1,5,9,13,14},
		{3,4,5,6,7,8},
		{1,2,6,10,14,15},
		{7,8,9,10,11,12},
		{1,2,5,9,12,13},
		{0,4,5,6,7,11},
		{2,3,6,10,13,14},
		{4,8,9,10,11,15},

		{0,1,5,8,9,13},
		{1,3,4,5,6,7},
		{2,6,7,10,14,15},
		{8,9,10,11,12,14},
		{1,4,5,9,12,13},
		{0,2,4,5,6,7},
		{2,3,6,10,11,14},
		{8,9,10,11,13,15},

		{0,1,2,4,6,10},
		{2,3,7,9,10,11},
		{5,9,11,13,14,15},
		{4,5,6,8,12,13},
		{2,6,8,10,12,14},
		{0,1,4,8,9,10},
		{1,2,3,5,7,9},
		{5,6,7,11,14,15}
	};
	/*
	int network_index[32][6]={
		
		{0,1,2,3,4,5},
		{4,5,6,7,8,9},
		{5,6,7,9,10,11},
		{9,10,11,13,14,15},

		{2,3,6,7,11,15},
		{1,2,5,6,10,14},
		{5,6,9,10,13,14},
		{4,5,8,9,12,13},

		{11,15,10,14,13,12},
		{7,11,6,10,9,8},
		{4,5,6,8,9,10},
		{0,1,2,4,5,6},
		{12,13,8,9,4,0},
		{13,14,9,10,5,1},
		{9,10,5,6,1,2},
		{10,11,6,7,2,3},
		
		{12,13,14,15,8,9},
		{8,9,10,11,4,5},
		{9,10,11,5,6,7},
		{5,6,7,1,2,3},
		
		{0,4,8,12,1,5},
		{1,5,9,13,2,6},
		{5,6,9,10,13,14},
		{6,7,10,11,14,15},
		
		{0,1,2,3,6,7},
		{4,5,6,7,10,11},
		{4,5,6,8,9,10},
		{8,9,10,12,13,14},

		{15,11,7,3,14,10},
		{14,10,6,2,13,9},
		{13,14,9,10,5,6},
		{12,13,8,9,4,5}

	};
	*/

};

/**
 * default random environment, i.e., placer
 * place the hint tile and decide a new hint tile
 */
class random_placer : public random_agent {
public:
	random_placer(const std::string& args = "") : random_agent("name=place role=placer " + args) {
		spaces[0] = { 12, 13, 14, 15 };
		spaces[1] = { 0, 4, 8, 12 };
		spaces[2] = { 0, 1, 2, 3};
		spaces[3] = { 3, 7, 11, 15 };
		spaces[4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	}

	virtual action take_action(const board& after) {
		std::vector<int> space = spaces[after.last()];
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;

			int bag[3], num = 0;
			for (board::cell t = 1; t <= 3; t++)
				for (size_t i = 0; i < after.bag(t); i++)
					bag[num++] = t;
			std::shuffle(bag, bag + num, engine);

			board::cell tile = after.hint() ?: bag[--num];
			board::cell hint = bag[--num];

			return action::place(pos, tile, hint);
		}
		return action();
	}

private:
	std::vector<int> spaces[5];
};


/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class random_slider : public random_agent {
public:
	random_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};

class my_player : public agent {
public:
	my_player(const std::string& args = "") : agent(args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		//std::shuffle(opcode.begin(), opcode.end(), engine);
		board::reward best_reward = -1;
		int best_op = -1;
		//int best_count=-1;
		//int merge_op=-1;
		for (int f_op : opcode) {
			auto temp=board(before);
			board::reward f_reward = temp.slide(f_op);
			//int f_mergecount=temp.total_merge();
			int s_bestcount=-1;
			
			for(int s_op:opcode){
				auto s_temp=board(temp);
				board::reward s_reward=s_temp.slide(s_op);
				//int s_mergecount=s_temp.total+merge();
				if(s_bestcount<s_reward){
					s_bestcount=s_reward;
				}

			}

			if(f_reward+s_bestcount>best_reward){
				best_reward=f_reward+s_bestcount;
				best_op=f_op;
			}
		}
		if(best_reward!=-1){
			return action::slide(best_op);
		}
		else{
			return action();
		}
	}

private:
	std::array<int, 4> opcode;
};