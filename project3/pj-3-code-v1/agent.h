/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
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
#include <ctime> 
#include "board.h"
#include "action.h"

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
	virtual int simulation_step() const{return std::stoi(property("simulation"));}
	virtual int duration() const{return std::stoi(property("timeout"));}

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
 * random player for both side
 * put a legal piece randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
	}

	virtual action take_action(const board& state) {
		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
		}
		return action();
	}

private:
	std::vector<action::place> space;
	board::piece_type who;
};


class mcts_player : public random_agent {
public:
	mcts_player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), myop_space(board::size_x*board::size_y), who(board::empty), opponent(board::empty){
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") {
			who = board::black;
			opponent=board::white;
			}
		if (role() == "white"){ 
			who = board::white;
			opponent=board::black;
			}
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
		for(size_t i =0;i<myop_space.size();i++){
			myop_space[i] = action::place(i,opponent);
		}
	}
	/*
	virtual action take_random_action(const board& state) {
		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
		}
		return action();
	}
	*/

	board::piece_type my_opponent(board::piece_type myself){
		if(myself== board::black){
			return board::white;
		}
		else{
			return board::black;
		}
	}

	struct node{	
			node *parent=nullptr;
			std::vector<node*> childrens;
			int wi=0;
			int si=0;
			bool isleaf=true;
			board state;
			board::piece_type self;
			action::place move_placed;
			bool is_terminal=false;
			node(node *p, board s, board::piece_type me){
				parent=p;
				state=s;
				self=me;	
			}

	};

			double best_child(node *child){
				if(who==child->self){
					if(child->si==0){
						return double(1000000);
					}
					else{
						return double(child->wi)/double(child->si)+exploration_c*sqrt(log(double(child->parent->si))/double(child->si));
					}
				}
				else{
					if(child->si==0){
						return double(1000000);
					}
					else{
						return (1.0-double(child->wi)/double(child->si))+exploration_c*sqrt(log(double(child->parent->si))/double(child->si));
					}
				}
			}

			node *selection(node *current){
				while(!current->isleaf){
					float best=0;
					int best_index=0;
					for(int i=0;i<current->childrens.size();i++){
						if(best<best_child(current->childrens[i])){
							best=best_child(current->childrens[i]);
							best_index=i;
						}
					}
					current=current->childrens[best_index];
				}
				//std::cout<<"selecting child................"<<std::endl;
				return current;
			}

			node* expand(node *current){
				if(current->is_terminal){
					return current;
				}
				else{
					board::piece_type op=my_opponent(current->self);
					if(op==who){
						for(int i=0;i<space.size();i++){
							board temp=current->state;
							board t=current->state;
							action::place placement(i,op);
							if(placement.apply(t)==board::legal){
								placement.apply(temp);
								node* newnode=new node(current,temp,op);
								newnode->move_placed=placement;
								current->childrens.push_back(newnode);
							}
						}
					}
					else{
						for(int i=0;i<myop_space.size();i++){
							board temp=current->state;
							board t=current->state;
							action::place placement(i,op);
							if(placement.apply(t)==board::legal){
								placement.apply(temp);
								node* newnode=new node(current,temp,op);
								newnode->move_placed=placement;
								current->childrens.push_back(newnode);
							}
						}
					}

					//std::cout<<"expand child"<<std::endl;
					if(current->childrens.empty()){
						//std::cout<<"this child is terminal........."<<std::endl;
						current->is_terminal=true;
						return current;
					}
					else{
						std::shuffle(current->childrens.begin(),current->childrens.end(),engine);
						current->isleaf=false;
						return current->childrens.front();
					}
				}
			}

			int rollout(node* current,std::vector<action::place>* my_space, std::vector<action::place>* op_space){
				
				board::piece_type myop=my_opponent(current->self);
				board::piece_type next;
					
				std::vector<action::place>* current_space;

				std::shuffle(my_space->begin(),my_space->end(),engine);
				std::shuffle(op_space->begin(),op_space->end(),engine);

				//std::cout<<"simulation..."<<std::endl;
				board temp=current->state;
			
				int score=0;
				if(current->is_terminal){
					score=(who!=myop);
					return score;
				}
				
				while(1){		
					int terminate=1;
					if(myop==who){
						current_space=my_space;
						next=opponent;
					}
					else{
						current_space=op_space;
						next=who;
					}
					
					for (const action::place& move : *(current_space)) {
						board t=temp;
						if (move.apply(t) == board::legal){
							move.apply(temp);
							terminate = 0;
							break;
						}
					}
					
					if(terminate==1){
						if(myop==who){
							score=0;
						}
						else{
							score=1;
						}
						break;
					}
					myop=next;
				}
				return score;

			}

			void backpropogation(node *current, int score){
				while(current!=nullptr){				
					current->wi+=score;
					current->si++;
					current=current->parent;
				}
				//std::cout<<"backpropogate"<<std::endl;
			}

	void deletenode(node * current){
		for(int i=0;i<current->childrens.size();i++){
			deletenode(current->childrens[i]);
		}
		delete current;
	}


	virtual action take_action(const board& state) {
		action::place best_move=action();
		int simulation_count=0;
		int time_duration=0;
				
		node *root=new node(nullptr,state, opponent);

		start=clock();

		std::vector<action::place> my_space=space;
		std::vector<action::place> opponent_space=myop_space;

		while(1){
			simulation_count++;
			node *best_leaf=selection(root);
			node *new_leaf=expand(best_leaf);
			int score=rollout(new_leaf, &my_space, &opponent_space);
			backpropogation(new_leaf,score);

			if(simulation_count%500==0){
				end=clock();
				if((end-start)/CLOCKS_PER_SEC>0.999999){
					break;
				}
			}

		}
		//std::cout<<"choose moves"<<std::endl;
		int bestcount=-1;
		
		for(int i=0;i<root->childrens.size();i++){
			if(root->childrens[i]->si>bestcount){
				bestcount=root->childrens[i]->si;
				best_move=root->childrens[i]->move_placed;
			}
		}
		
		
		deletenode(root);
		return best_move;
	}

private:
	std::vector<action::place> space;
	std::vector<action::place> myop_space;
	board::piece_type who;
	board::piece_type opponent;
	float exploration_c=0.8;
	clock_t start,end;
};

