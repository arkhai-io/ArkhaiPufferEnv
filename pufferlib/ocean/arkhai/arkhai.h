#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include "raylib.h"

#define MAX_JOBS 100

#define NUM_ACT 2
#define MAX_NODE_TYPES 16

// Whenever you call vec_log, PufferLib will
// sum all fields in Log across all env instances per-core
// and divide by n. All fields must be floats, and n must
// be the last field.
typedef struct {
    float score;
    float expense;
    float profit;
    float episode_length;
    float episode_return;
    float n;
} Log;

typedef struct {
    int nodes[MAX_NODE_TYPES];
    float tb_usage;
    int duration;
    int start;
    bool active;
    float price; // Per tick
    int negotiations;
    float efficiency; // Should be a joint property of job and cluster...hard to model
} Job;

typedef struct {
    int total;
    int free;
} Node;

typedef struct {
    Node nodes[MAX_NODE_TYPES];
    float tb_capacity;
    float tb_usage;
    float kwh_storage;
    float kwh_capacity;
    float kw_generation;
    // Placeholders are here but these are not really meaningful without some
    // sort of requirement or reward directive on job requests
    int sla;
    int location;
    float uptime;
    float latency;
    float bandwidth;
    float reputation;
} Cluster;

// Different clusters can have different specifications with noise.
// Clusters are sampled from a ClusterSpec params + domain randomization (dr).
typedef struct {
    int node_capacity[MAX_NODE_TYPES];
    float node_capacity_dr[MAX_NODE_TYPES];
    int tb_capacity;
    float tb_capacity_dr;
    float kwh_capacity;
    float kwh_capacity_dr;
    float kw_generation;
    float kw_generation_dr;
} ClusterSpec;

// Agents can be buyers or sellers, heuristic or RL
typedef struct {
    ClusterSpec cluster_spec;
    Cluster cluster;
    Job jobs[MAX_JOBS];
    Job request;
    float job_revenue;
    float compute_expense;
    float energy_revenue;
    float energy_expense;
    float prev_reward;
    float episode_return;
    int filled_jobs;
    bool is_heuristic;
    bool is_buyer;
} Agent;

typedef struct {
    // This chunk is all required puffer api
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    // ... until here. These pointers get set to chunks of
    // shared memory by env_binding.h
    Agent* agents;
    int ai_sellers;
    int ai_buyers;
    int scripted_buyers;
    int scripted_sellers;
    int num_agents;
    int tick;
    int episode_length;
    int request_timeout;
    int node_types;
    int num_obs;
    int job_nodes[MAX_NODE_TYPES];
    float job_nodes_dr[MAX_NODE_TYPES];
    int job_duration;
    float job_duration_dr;
    float job_tb_usage;
    float job_tb_usage_dr;
    float job_efficiency;
    float job_efficiency_dr;
    float scripted_buy_price;
    float scripted_buy_price_dr;
    float scripted_sell_price;
    float scripted_sell_price_dr;
    float reward_scale;
    float tb_price;
    float node_prices[MAX_NODE_TYPES];
    float node_energy_kw[MAX_NODE_TYPES];
    float energy_demand_base;
    float kwh_price_base;
    float kwh_price_sensitivity;
    float kwh_demand_threshold;
    float a1;
    float b1;
    float a2;
    float b2;
    float a3;
    float b3;
    int randomize_offset;
    int preset;
    int serving;
    bool debug;
} Arkhai;

// I was playing with having "presets" for different training configs.
// It's a bit fiddly though. There are way more env knobs than when I introduced this.
enum PRESET {
    NONE,
    DEFAULT,
    ENERGY_PRODUCER,
    STORAGE_CENTER,
    PREMIUM_HPC,
};

enum SIDE {
    SELLER,
    BUYER,
    BOTH,
};

float randf(float min, float max) {
    return min + ((float)rand()/(float)(RAND_MAX))*(max-min);
}

// Simple uniform randomization with dr
float randomized(float base, float dr) {
    return randf(base*(1.0f-dr), base*(1.0f+dr));
}

void init_cluster(Cluster* cluster, ClusterSpec* spec, int node_types) {
    cluster->kw_generation = randomized(spec->kw_generation, spec->kw_generation_dr);
    cluster->kwh_capacity = randomized(spec->kwh_capacity, spec->kwh_capacity_dr);
    cluster->tb_capacity = randomized(spec->tb_capacity, spec->tb_capacity_dr);
    for (int i=0; i<node_types; i++) {
        int capacityy = spec->node_capacity[i];
        float capacity_dr = spec->node_capacity_dr[i];
        cluster->nodes[i].total = randomized(capacityy, capacity_dr);
        cluster->nodes[i].free = cluster->nodes[i].total;
    }
}

/*
void apply_kwh_storage_producer_preset(ClusterSpec* spec) {
    env->max_nodes = 0;
    env->max_tb_usage = 0;
    env->max_kw_generation = 100;
    env->max_kwh_capacity = 1000;
}

void apply_storage_center_preset(Arkhai* env) {
    env->max_nodes = 0;
    env->max_kw_generation = 0;
    env->max_kwh_capacity = 0;
    env->max_tb_usage = 10000;
    env->tb_usage_price = 0.02;
}

// TODO: add sla, rep, etc
void apply_premium_hpc_preset(Arkhai* env) {
    for (int i = 0; i < env->node_types; i++) {
        env->node_prices[i] *= 1.2;
    }
    env->tb_usage_price *= 1.2;
}
*/

void init(Arkhai* env, ClusterSpec buyer_spec, ClusterSpec seller_spec) {
    int num_buyers = env->ai_buyers + env->scripted_buyers;
    assert(num_buyers > 0);

    int num_sellers = env->ai_sellers + env->scripted_sellers;
    assert(num_sellers > 0);
    
    env->num_agents = num_buyers + num_sellers;
    env->agents = calloc(env->num_agents, sizeof(Agent));

    // Note storage order: RL agents first, then heuristic agents
    int agent_idx = 0;
    for (int i=0; i<env->ai_sellers; i++) {
        env->agents[agent_idx].cluster_spec = seller_spec;
        env->agents[agent_idx].is_heuristic = false;
        env->agents[agent_idx].is_buyer = false;
        agent_idx++;
    }
    for (int i=0; i<env->ai_buyers; i++) {
        env->agents[agent_idx].cluster_spec = buyer_spec;
        env->agents[agent_idx].is_heuristic = false;
        env->agents[agent_idx].is_buyer = true;
        agent_idx++;
    }
    for (int i=0; i<env->scripted_sellers; i++) {
        env->agents[agent_idx].cluster_spec = seller_spec;
        env->agents[agent_idx].is_heuristic = true;
        env->agents[agent_idx].is_buyer = false;
        agent_idx++;
    }
    for (int i=0; i<env->scripted_buyers; i++) {
        env->agents[agent_idx].cluster_spec = buyer_spec;
        env->agents[agent_idx].is_heuristic = true;
        env->agents[agent_idx].is_buyer = true;
        agent_idx++;
    }

    assert(env->node_types > 0 && env->node_types <= MAX_NODE_TYPES);

    // Sanity checks. These are here because it is easy to mess up init
    assert(env->ai_sellers >= 0);
    assert(env->ai_buyers >= 0);
    assert(env->scripted_buyers >= 0);
    assert(env->scripted_sellers >= 0);
    assert(env->num_agents > 0);
    assert(env->episode_length > 0);
    assert(env->request_timeout >= 0);
    assert(env->job_duration > 0);
    assert(env->job_duration_dr >= 0.0f);
    assert(env->job_tb_usage >= 0);
    assert(env->job_tb_usage_dr >= 0.0f);
    assert(env->scripted_sell_price > 0.0f);
    assert(env->scripted_sell_price_dr >= 0.0f);
    assert(env->scripted_buy_price > 0.0f);
    assert(env->scripted_buy_price_dr >= 0.0f);
    assert(env->reward_scale > 0.0f);
    assert(env->tb_price >= 0.0f);
    for (int i = 0; i < env->node_types; i++) {
        assert(env->node_prices[i] >= 0.0f);
        assert(env->node_energy_kw[i] > 0.0f);
    }
    assert(env->energy_demand_base >= 0.0f);
    assert(env->kwh_price_base >= 0.0f);
    assert(env->kwh_price_sensitivity >= 0.0f);
    assert(env->kwh_demand_threshold >= 0.0f);

    assert(env->randomize_offset == 0 || env->randomize_offset == 1);
    for (int agent_idx=0; agent_idx<env->num_agents; agent_idx++) {
        ClusterSpec* spec = &env->agents[agent_idx].cluster_spec;
        for (int i=0; i<env->node_types; i++) {
            assert(spec->node_capacity[i] >= 0);
            assert(spec->node_capacity_dr[i] >= 0.0f);
        }
        assert(spec->tb_capacity >= 0);
        assert(spec->tb_capacity_dr >= 0.0f);
        assert(spec->kwh_capacity >= 0);
        assert(spec->kwh_capacity_dr >= 0.0f);
        assert(spec->kw_generation >= 0);
        assert(spec->kw_generation_dr >= 0.0f);
    }

}

// Used to select a random buy order
int select_buyer(Arkhai* env) {
    int num_buyers = env->scripted_buyers + env->ai_buyers;
    int idxs[num_buyers];
    int i = 0;
    for (int agent_idx=0; agent_idx<env->num_agents; agent_idx++) {
        Agent* agent = env->agents + agent_idx;
        if (agent->is_buyer) {
            idxs[i++] = agent_idx;
        }
    }

    int idx_idx = rand()%num_buyers;
    return idxs[idx_idx];
}


void sanity_check(Job job, int node_types) {
    assert(job.tb_usage >= 0);
    for (int i=0; i<node_types; i++) {
        assert(job.nodes[i] >= 0);
    }
}

float job_price(Arkhai* env, Job* job) {
    float price = env->tb_price*job->tb_usage;
    for (int i=0; i<env->node_types; i++) {
        price += env->node_prices[i]*job->nodes[i];
    }
    return price;
}

// Random job request, based on job spec settings. If no jobs are
// filling, check that your seller has enough capacity
Job generate_request(Arkhai* env) {
    Job job = (Job) {
        .tb_usage = randomized(env->job_tb_usage, env->job_tb_usage_dr),
        .duration = randomized(env->job_duration, env->job_duration_dr),
        .start = env->tick,
        .active = true,
        .negotiations = 0,
    };
    for (int i=0; i<env->node_types; i++) {
        int nodes = env->job_nodes[i];
        float nodes_dr = env->job_nodes_dr[i];
        job.nodes[i] = randomized(nodes, nodes_dr);
    }
    job.price = job_price(env, &job);
    return job;
}

void compute_observations(Arkhai* env) {
    int i = 0;

    // DO NOT ADD OR CHANGE INDEXING WITHOUT UPDATING BOTH BUYER AND SELLER CODE,
    // AS WELL AS THE OBS SIZE IN PYTHON. WE DO NOT HAVE A GOOD WAY TO AUTOMATICALLY
    // CHECK THIS. YOU WILL CAUSE SILENT MEMORY CORRUPTION OR SEGFAULTS.
    Job* request = &env->agents[env->serving].request;
    int num_agents = env->ai_sellers + env->ai_buyers;
    for (int agent_idx=0; agent_idx<num_agents; agent_idx++) {
        Agent* agent = env->agents + agent_idx;
        Cluster* cluster = &agent->cluster;

        env->observations[i++] = (env->tick % 24) / 24.0f;
        for (int j=0; j<env->node_types; j++) {
            env->observations[i++] = cluster->nodes[j].total / ((float)env->job_nodes[j] + 1);
            env->observations[i++] = cluster->nodes[j].free / ((float)env->job_nodes[j] + 1);
        }
        env->observations[i++] = cluster->tb_usage / ((float)env->job_tb_usage + 1);
        env->observations[i++] = cluster->tb_capacity / ((float)env->job_tb_usage + 1);
        // TODO: these should be divided by global capacities
        env->observations[i++] = cluster->kwh_storage / ((float)cluster->kwh_capacity + 1);
        env->observations[i++] = cluster->kwh_capacity / ((float)cluster->kwh_capacity + 1);
        env->observations[i++] = cluster->kw_generation / ((float)cluster->kwh_capacity + 1);
        for (int j=0; j<env->node_types; j++) {
            env->observations[i++] = request->nodes[j] / ((float)env->job_nodes[j] + 1);
        }
        env->observations[i++] = request->tb_usage / ((float)env->job_tb_usage + 1);
        env->observations[i++] = request->start / ((float)env->job_duration + 1);
        env->observations[i++] = request->duration / ((float)env->job_duration + 1);
        env->observations[i++] = request->negotiations / ((float)env->request_timeout + 1);
        env->observations[i++] = request->price / (job_price(env, request) + 1);
        env->observations[i++] = agent->prev_reward;
    }
}

void c_reset(Arkhai* env) {
    // This is for first-time reset only. Staggering improves training stability.
    // Forgetting to disable this is a great way to break your evals.
    if (env->randomize_offset && env->tick == 0) {
        env->tick = 0 + rand()%env->episode_length;
    } else {
        env->tick = 0;
    }

    for (int agent_idx=0; agent_idx<env->num_agents; agent_idx++) {
        Agent* agent = env->agents + agent_idx;
        agent->job_revenue = 0.0f;
        agent->compute_expense = 0.0f;
        agent->energy_revenue = 0.0f;
        agent->energy_expense = 0.0f;
        agent->prev_reward = 0.0f;
        agent->episode_return = 0.0f;
        agent->filled_jobs = 0;
        init_cluster(&agent->cluster, &agent->cluster_spec, env->node_types);
        memset(agent->jobs, 0, MAX_JOBS*sizeof(Job));
        if (agent->is_buyer) {
            agent->request = generate_request(env);
        }
    }
    env->serving = select_buyer(env);
    compute_observations(env);
}

bool can_accept_job(Arkhai* env, Agent* agent, Job* job) {
    Cluster* cluster = &agent->cluster;
    if (job->tb_usage > cluster->tb_capacity) {
        return false;
    }
    for (int j=0; j<env->node_types; j++) {
        if (job->nodes[j] > cluster->nodes[j].free) {
            return false;
        }
    }
    for (int i=0; i<MAX_JOBS; i++) {
        if (agent->jobs[i].active) {
            continue;
        }
        return true;
    }
    return false;
}

// Mark nodes/tb as used on accept
void accept_job(Arkhai* env, Job* job, int idx) {
    assert(idx >= 0 && idx < env->num_agents);
    Agent* agent = &env->agents[idx];
    Cluster* cluster = &agent->cluster;
    for (int i=0; i<MAX_JOBS; i++) {
        if (agent->jobs[i].active) {
            continue;
        }
        agent->jobs[i] = *job;
        for (int j=0; j<env->node_types; j++) {
            cluster->nodes[j].free -= job->nodes[j];
        }
        //printf("Accepted job on tick %d . Remaining gpu_2 nodes: %d\n", env->tick, cluster->nodes[2].free);
        cluster->tb_capacity -= job->tb_usage;
        return;
    }
}

// We assume jobs can have an efficiency modifier. You should set
// job_efficiency < 1 to compensate if you use this.
float job_kw(Arkhai* env, Job job) {
    float kw = 0.0f;
    for (int i=0; i<env->node_types; i++) {
        kw += job.nodes[i]*env->node_energy_kw[i];
    }
    float efficiency = randomized(env->job_efficiency, env->job_efficiency_dr);
    return efficiency*kw;
}
 
// These two fns are simple responses for single-side sims. Ideally, you'd match them to
// the distribution of real-world demand to first order + heavy randomization.
float buyer_response(Arkhai* env, Job request, float offer_price) {
    float rng = randomized(env->scripted_buy_price, env->scripted_buy_price_dr);
    return rng * job_price(env, &request);
}

float seller_response(Arkhai* env, Job request, float offer_price) {
    float rng = randomized(env->scripted_sell_price, env->scripted_sell_price_dr);
    return rng * job_price(env, &request);
}

float calculate_price(float demand, float p0, float threshold, float c) {
    float excess = fmaxf(0.0f, demand - threshold);
    return p0 + c*powf(excess, 2.0f); // Quadratic for non-linear spike
}

// I didn't know how you would want to price energy so I just gave you
// some Fourier components and fit something that looks roughly like the
// wikipedia graphs.
float kw_price(Arkhai* env, float t) {
    float demand = env->energy_demand_base + (
        env->a1*cosf(2.0f*PI*t/24.0f) + env->b1*sinf(2.0f*PI*t/24.0f) +
        env->a2*cosf(4.0f*PI*t/24.0f) + env->b2*sinf(4.0f*PI*t/24.0f) +
        env->a3*cosf(6.0f*PI*t/24.0f) + env->b3*sinf(6.0f*PI*t/24.0f));
    float excess = fmaxf(0.0f, demand - env->kwh_demand_threshold);
    return env->kwh_price_base + env->kwh_price_sensitivity*powf(excess, 2.0f);
}

// Clear/update fns will bottleneck perf eventually with enough agents/jobs
// and we will have to do something smarter, but it is fast for now.
void clear_finished_jobs(Arkhai* env) {
    for (int agent_idx=0; agent_idx<env->num_agents; agent_idx++) {
        Agent* agent = env->agents + agent_idx;
        Cluster* cluster = &agent->cluster;
        for (int i=0; i<MAX_JOBS; i++) {
            Job job = agent->jobs[i];
            if (!job.active) {
                continue;
            }
            if (env->tick < job.start + job.duration) {
                continue;
            }
            for (int j=0; j<env->node_types; j++) {
                cluster->nodes[j].free += job.nodes[j];
            }
            cluster->tb_capacity += job.tb_usage;
            memset(&agent->jobs[i], 0, sizeof(Job));
        }
    }
}

void update_jobs(Arkhai* env) {
    for (int agent_idx=0; agent_idx<env->num_agents; agent_idx++) {
        Agent* agent = env->agents + agent_idx;
        Cluster* cluster = &agent->cluster;
        for (int i=0; i<MAX_JOBS; i++) {
            if (agent->is_buyer) {
                continue;
            }

            Job job = agent->jobs[i];
            if (!job.active) {
                continue;
            }

            float kw = job_kw(env, job);
            if (cluster->kwh_storage > kw) {
                cluster->kwh_storage -= kw;
                kw = 0;
            } else {
                kw -= cluster->kwh_storage;
                cluster->kwh_storage = 0;
            }

            // Jobs incur energy expense as it is used. Pricing is upfront.
            float energy_expense = kw*kw_price(env, env->tick);
            agent->energy_expense += energy_expense;
        }
    }
} 

void c_step(Arkhai* env) {
    int ai_agents = env->ai_sellers + env->ai_buyers;
    memset(env->rewards, 0, ai_agents*sizeof(float));
    memset(env->terminals, 0, ai_agents*sizeof(unsigned char));

    // We have an episode timeout. You need this to resample domain randomization.
    // It can also mask degenerate states though. We should test some longer runs.
    if (env->tick >= env->episode_length) {
        for (int agent_idx=0; agent_idx<env->num_agents; agent_idx++) {
            Agent* agent = env->agents + agent_idx;
            if (env->debug) {
                printf("Agent %d\n", agent_idx);
                printf("\tIs Heuristic: %d\n", agent->is_heuristic);
                printf("\tIs Buyer: %d\n", agent->is_buyer);
                printf("\tJob Revenue: %f\n", agent->job_revenue);
                printf("\tCompute Expense: %f\n", agent->compute_expense);
                printf("\tFilled Jobs: %d\n", agent->filled_jobs);
                printf("\tEnergy Revenue: %f\n", agent->energy_revenue);
                printf("\tEnergy Expense: %f\n", agent->energy_expense);
                printf("\tEpisode Return: %f\n", agent->episode_return);
            }

            if (agent->is_heuristic) {
                continue;
            }

            float job_profit = agent->job_revenue - agent->compute_expense;
            float energy_profit = agent->energy_revenue - agent->energy_expense;
            env->log.profit += job_profit + energy_profit;
            env->log.expense += agent->compute_expense + agent->energy_expense;
            env->log.score += job_profit + energy_profit;
            env->log.episode_length += env->tick;
            env->log.episode_return += agent->episode_return;
            env->log.n++;
            env->terminals[agent_idx] = 1;
        }
        c_reset(env);
        return;
    }

    int serving = env->serving;
    Job* request = &env->agents[serving].request;
    request->negotiations++;
    
    assert(request != NULL);
    float base_price = job_price(env, request);
    int request_idx = serving;

    Agent* buyer = &env->agents[request_idx];
    float request_price;
    if (buyer->is_heuristic) {
        request_price = buyer_response(env, *request, base_price);
    } else {
        // I am discretizing the action space to +/- 20% of base price estimate in buckets of 5%
        // This does not need to be uniform, and we can adjust it for finer pricing. Not too
        // many buckets though.
        // -0.2 -0.15 -0.1 -0.05 0.0f 0.05 0.1 0.15 0.2
        float price_mul = 1.0f + ((float)env->actions[2*request_idx] - 4.0f)/20.0f;
        request_price = price_mul * base_price;
    }

    // There can be multiple buyers, but sellers only make offers on one job at a time.
    int best_idx = -1;
    float best_response_price = FLT_MAX;
    bool exists_valid_buyer = false; // We skip negotiation if nobody has capacity
    Agent* seller;
    for (int agent_idx=0; agent_idx<env->num_agents; agent_idx++) {
        if (agent_idx == request_idx) {
            continue;
        }
        seller = &env->agents[agent_idx];
        if (!can_accept_job(env, seller, request)) {
            continue;
        }
        exists_valid_buyer = true;
        float offer_price;
        if (seller->is_heuristic) {
            offer_price = seller_response(env, *request, base_price);
        } else {
            float price_mul = 1.0f + ((float)env->actions[2*agent_idx] - 4.0f)/20.0f;
            offer_price = price_mul * base_price;
        }
        if (offer_price <= best_response_price) {
            best_idx = agent_idx;
            best_response_price = offer_price;
        }
    }
    if (!exists_valid_buyer) {
        // Skip negotiation
    } else if (best_response_price <= request_price) {
        // The best seller offer wins the job
        accept_job(env, request, best_idx);

        // Full duration used for reward. Clipped duration used for logs.
        // This prevents the agent from exploiting terminal bounds...
        // Somewhat. It gets to "ignore" energy expense past the end of the episode.
        // We should probably come up with a way around this. But energy expense is
        // actually not that high. It looks like most of pricing is capex on chips.
        float duration = request->duration;
        float clipped_duration = duration;
        if (env->episode_length - env->tick < duration) {
            clipped_duration = env->episode_length - env->tick;
        }

        float buyer_revenue = job_price(env, request);
        float buyer_expense = best_response_price;
        buyer->job_revenue += buyer_revenue*clipped_duration;
        buyer->compute_expense += buyer_expense*clipped_duration;
        buyer->filled_jobs++;
        if (!buyer->is_heuristic) {
            env->rewards[request_idx] += (buyer_revenue - buyer_expense)*duration;
        }

        seller = &env->agents[best_idx];
        float seller_revenue = best_response_price;
        seller->job_revenue += seller_revenue*clipped_duration;
        //printf("recognizing seller revenue %f\n", seller_revenue*clipped_duration);
        // Energy is recognized as it is incurred.

        seller->filled_jobs++;
        if (!seller->is_heuristic) {
            env->rewards[best_idx] += seller_revenue*duration;
        }
        buyer->request = generate_request(env);
    } else if (request->negotiations < env->request_timeout) {
        compute_observations(env);
        return;
    }

    env->tick++;
    clear_finished_jobs(env);
    buyer->request = generate_request(env);

    // Assume energy over capacity is sold at the current market price
    for (int agent_idx=0; agent_idx<env->num_agents; agent_idx++) {
        Agent* agent = &env->agents[agent_idx];
        Cluster* cluster = &agent->cluster;
        cluster->kwh_storage += cluster->kw_generation;
        if (cluster->kwh_storage <= cluster->kwh_capacity) {
            continue;
        }
        float diff = cluster->kwh_storage - cluster->kwh_capacity;
        cluster->kwh_storage = cluster->kwh_capacity;
        float energy_revenue = diff*kw_price(env, env->tick);
        agent->energy_revenue += energy_revenue;
        if (!agent->is_heuristic) {
            env->rewards[agent_idx] += energy_revenue;
        }
    }

    update_jobs(env);

    for (int agent_idx=0; agent_idx<ai_agents; agent_idx++) {
        Agent* agent = &env->agents[agent_idx];
        Cluster* cluster = &agent->cluster;

        // Sell energy
        if (env->actions[2*agent_idx + 1] > 0) {
            float amt = 0.5f * cluster->kwh_storage;
            float profit = amt*kw_price(env, env->tick);
            cluster->kwh_storage -= amt;
            agent->energy_revenue += profit;
            env->rewards[agent_idx] += profit;
        }
     
        // Scale rewards
        float reward = env->rewards[agent_idx];
        reward *= env->reward_scale;
        env->rewards[agent_idx] = reward;
        agent->episode_return += reward;
        agent->prev_reward = reward;
    }

    compute_observations(env);
}

// Placeholder. There isn't a renderer at the moment.
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

void c_render(Arkhai* env) {
    if (!IsWindowReady()) {
        InitWindow(1080, 720, "PufferLib Arkhai");
        SetTargetFPS(5);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    EndDrawing();
}

void c_close(Arkhai* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    free(env->agents);
}
