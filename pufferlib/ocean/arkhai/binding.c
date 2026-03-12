#include "arkhai.h"

#define Env Arkhai 
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->node_types = (int)unpack(kwargs, "node_types");
    env->num_obs = 12 + 3 * env->node_types;
    env->ai_sellers = unpack(kwargs, "ai_sellers");
    env->ai_buyers = unpack(kwargs, "ai_buyers");
    env->scripted_sellers = unpack(kwargs, "scripted_sellers");
    env->scripted_buyers = unpack(kwargs, "scripted_buyers");
    env->episode_length = unpack(kwargs, "episode_length");
    env->request_timeout = unpack(kwargs, "request_timeout");
    char key[64];
    for (int i = 0; i < env->node_types; i++) {
        snprintf(key, sizeof(key), "job_gpu_%d_nodes", i);
        env->job_nodes[i] = unpack(kwargs, key);
        snprintf(key, sizeof(key), "job_gpu_%d_nodes_dr", i);
        env->job_nodes_dr[i] = unpack(kwargs, key);
    }
    env->job_duration = unpack(kwargs, "job_duration");
    env->job_duration_dr = unpack(kwargs, "job_duration_dr");
    env->job_tb_usage = unpack(kwargs, "job_tb_usage");
    env->job_tb_usage_dr = unpack(kwargs, "job_tb_usage_dr");
    env->job_efficiency = unpack(kwargs, "job_efficiency");
    env->job_efficiency_dr = unpack(kwargs, "job_efficiency_dr");
    env->scripted_buy_price = unpack(kwargs, "scripted_buy_price");
    env->scripted_buy_price_dr = unpack(kwargs, "scripted_buy_price_dr");
    env->scripted_sell_price = unpack(kwargs, "scripted_sell_price");
    env->scripted_sell_price_dr = unpack(kwargs, "scripted_sell_price_dr");
    env->reward_scale = unpack(kwargs, "reward_scale");
    env->tb_price = unpack(kwargs, "tb_price");
    for (int i = 0; i < env->node_types; i++) {
        snprintf(key, sizeof(key), "gpu_%d_price", i);
        env->node_prices[i] = unpack(kwargs, key);
        snprintf(key, sizeof(key), "gpu_%d_kw", i);
        env->node_energy_kw[i] = unpack(kwargs, key);
    }
    env->energy_demand_base = unpack(kwargs, "energy_demand_base");
    env->kwh_price_base = unpack(kwargs, "kwh_price_base");
    env->kwh_price_sensitivity = unpack(kwargs, "kwh_price_sensitivity");
    env->kwh_demand_threshold = unpack(kwargs, "kwh_demand_threshold");
    env->a1 = unpack(kwargs, "a1");
    env->b1 = unpack(kwargs, "b1");
    env->a2 = unpack(kwargs, "a2");
    env->b2 = unpack(kwargs, "b2");
    env->a3 = unpack(kwargs, "a3");
    env->b3 = unpack(kwargs, "b3");
    env->randomize_offset = unpack(kwargs, "randomize_offset");
    env->preset = unpack(kwargs, "preset");
    ClusterSpec seller_spec = {0};
    for (int i = 0; i < env->node_types; i++) {
        snprintf(key, sizeof(key), "cluster_gpu_%d_capacity", i);
        seller_spec.node_capacity[i] = unpack(kwargs, key);
        snprintf(key, sizeof(key), "cluster_gpu_%d_capacity_dr", i);
        seller_spec.node_capacity_dr[i] = unpack(kwargs, key);
    }
    seller_spec.tb_capacity = unpack(kwargs, "cluster_tb_capacity");
    seller_spec.tb_capacity_dr = unpack(kwargs, "cluster_tb_capacity_dr");
    seller_spec.kwh_capacity = unpack(kwargs, "cluster_kwh_capacity");
    seller_spec.kwh_capacity_dr = unpack(kwargs, "cluster_kwh_capacity_dr");
    seller_spec.kw_generation = unpack(kwargs, "cluster_kw_generation");
    seller_spec.kw_generation_dr = unpack(kwargs, "cluster_kw_generation_dr");
    ClusterSpec buyer_spec = {0};
    init(env, buyer_spec, seller_spec);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "expense", log->expense);
    assign_to_dict(dict, "profit", log->profit);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    return 0;
}
