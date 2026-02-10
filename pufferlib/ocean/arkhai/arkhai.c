#include "arkhai.h"

// Annoying: we have to dupe all the params from base. I can add a simple C ini
// parser in the next version. We have this for some tests in 4.0 and it works well.
Arkhai create_train_env() {
    return (Arkhai) {
        .tick=0,
        .node_types=3,
        .num_obs=21,
        .ai_sellers=1,
        .ai_buyers=0,
        .scripted_sellers=0,
        .scripted_buyers=1,
        .episode_length=100,
        .request_timeout=5,
        .job_nodes={10, 10, 10},
        .job_nodes_dr={0.2, 0.2, 0.2},
        .job_duration=10,
        .job_duration_dr=0.2,
        .job_tb_usage=0.2,
        .job_tb_usage_dr=0.2,
        .job_efficiency=0.8,
        .job_efficiency_dr=0.2,
        .scripted_buy_price=0.9,
        .scripted_buy_price_dr=0.2,
        .scripted_sell_price=0.9,
        .scripted_sell_price_dr=0.2,
        .reward_scale=0.0001,
        .tb_price=0.03,
        .node_prices={5.31, 15.92, 0.37},
        .node_energy_kw={6.5, 10.0, 1.0},
        .energy_demand_base=1500.0,
        .kwh_price_base=0.02,
        .kwh_price_sensitivity=0.0000001,
        .kwh_demand_threshold=1400,
        .a1=-374,
        .b1=-387,
        .a2=-4.6,
        .b2=-17.1,
        .a3=3.2,
        .b3=18.9,
        .debug=false,
    };
}

ClusterSpec create_train_spec() {
    return (ClusterSpec) {
        .node_capacity = {100, 100, 100},
        .node_capacity_dr = {0.2, 0.2, 0.2},
        .tb_capacity = 100,
        .tb_capacity_dr = 0.2,
        .kwh_capacity = 100,
        .kwh_capacity_dr = 0.2,
        .kw_generation = 10,
        .kw_generation_dr = 0.2,
    };
}

Arkhai create_test_env() {
    return (Arkhai) {
        .tick=0,
        .node_types=3,
        .num_obs=21,
        .episode_length=100,
        .job_duration=10,
        .job_duration_dr=0.0,
        .request_timeout=5,
        .scripted_buy_price=1.0,
        .scripted_buy_price_dr=0.0,
        .scripted_sell_price=1.0,
        .scripted_sell_price_dr=0.0,
        .job_efficiency=1.0,
        .job_efficiency_dr=0.0,
        .job_tb_usage=0,
        .job_nodes={1, 1, 1},
        .job_nodes_dr={0.0, 0.0, 0.0},
        .reward_scale=0.0001,
        .tb_price=0.03,
        .node_prices={5, 5, 0},
        .node_energy_kw={6.5, 10.0, 1.0},
        .energy_demand_base=0.0,
        .kwh_price_base=0.0,
        .kwh_price_sensitivity=0.0,
        .kwh_demand_threshold=0,
        .a1=0,
        .b1=0,
        .a2=0,
        .b2=0,
        .a3=0,
        .b3=0,
        .randomize_offset=0,
        .preset=NONE,
        .ai_sellers=0,
        .ai_buyers=0,
        .scripted_sellers=1,
        .scripted_buyers=1,
        .debug=true,
    };
}

ClusterSpec create_test_spec() {
    return (ClusterSpec) {
        .node_capacity = {1, 1, 1},
        .node_capacity_dr = {0.0, 0.0, 0.0},
        .tb_capacity = 10000,
        .tb_capacity_dr = 0.0,
        .kwh_capacity = 1000,
        .kwh_capacity_dr = 0.0,
        .kw_generation = 100,
        .kw_generation_dr = 0.0,
    };
}
 
int main() {
    // Basic sanity: fill 10 jobs. Note: revenue gets recognized upfront,
    // so the expected output is 1000 instead of 960
    printf("Basic sanity check\n");
    Arkhai env = create_test_env();
    ClusterSpec seller_spec = create_test_spec();
    ClusterSpec buyer_spec = {0};
    int num_agents = env.ai_buyers + env.ai_sellers;
    init(&env, buyer_spec, seller_spec);
    env.observations = (float*)calloc(num_agents*env.num_obs, sizeof(float));
    env.actions = (int*)calloc(num_agents*NUM_ACT, sizeof(int));
    env.rewards = (float*)calloc(num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(num_agents, sizeof(unsigned char));
    c_reset(&env);
    for (int i=0; i<env.episode_length; i++){
        c_step(&env);
    }
    assert(env.agents[0].job_revenue == 1000.0f && "Failed basic sale check");
    c_step(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    printf("Passed basic sanity check\n\n");

    // Energy production with no storage
    printf("Energy production with no storage\n");
    env = create_test_env();
    env.kwh_price_base = 1.0f;
    seller_spec = create_test_spec();
    seller_spec.kw_generation = 1;
    memset(seller_spec.node_capacity, 0, sizeof(int)*env.node_types);
    seller_spec.kwh_capacity = 0;
    buyer_spec = (ClusterSpec){0};
    num_agents = env.ai_buyers + env.ai_sellers;
    init(&env, buyer_spec, seller_spec);
    env.observations = (float*)calloc(num_agents*env.num_obs, sizeof(float));
    env.actions = (int*)calloc(num_agents*NUM_ACT, sizeof(int));
    env.rewards = (float*)calloc(num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(num_agents, sizeof(unsigned char));
    c_reset(&env);
    for (int i=0; i<env.episode_length; i++){
        c_step(&env);
    }
    assert(env.agents[0].energy_revenue == env.episode_length && "Failed energy production check");
    c_step(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    printf("Passed energy production with no storage\n\n");

    // Bilateral agent negotiation
    printf("Bilateral agent negotiation\n");
    env = create_test_env();
    env.ai_sellers = 1;
    env.ai_buyers = 1;
    env.scripted_sellers = 0;
    env.scripted_buyers = 0;
    seller_spec = create_test_spec();
    buyer_spec = (ClusterSpec){0};
    num_agents = env.ai_buyers + env.ai_sellers;
    init(&env, buyer_spec, seller_spec);
    env.observations = (float*)calloc(num_agents*env.num_obs, sizeof(float));
    env.actions = (int*)calloc(num_agents*NUM_ACT, sizeof(int));
    env.rewards = (float*)calloc(num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(num_agents, sizeof(unsigned char));
    c_reset(&env);
    for (int i=0; i<2*env.episode_length; i++){
        if (i%2 == 0) {
            env.actions[0] = 4; // Seller offers midpoint
            env.actions[2] = 2; // Buyer offers very low
        } else {
            env.actions[0] = 3; // Seller offers discount
            env.actions[2] = 3; // Buyer matches
        }
        c_step(&env);
    }
    assert(env.agents[0].job_revenue == 950.0f && "Bilateral negotiation seller incorrect revenue");
    assert(env.agents[1].job_revenue == 1000.0f && "Bilateral negotiation buyer incorrect revenue");
    assert(env.agents[1].compute_expense == 950.0f && "Bilateral negotiation buyer incorrect expense");
    c_step(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    printf("Passed bilateral agent negotiation\n\n");

    // Training environment
    printf("Mirrored training environment\n");
    env = create_train_env();
    seller_spec = create_train_spec();
    buyer_spec = (ClusterSpec){0};
    num_agents = env.ai_buyers + env.ai_sellers;
    init(&env, buyer_spec, seller_spec);
    env.observations = (float*)calloc(num_agents*env.num_obs, sizeof(float));
    env.actions = (int*)calloc(num_agents*NUM_ACT, sizeof(int));
    env.rewards = (float*)calloc(num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(num_agents, sizeof(unsigned char));
    c_reset(&env);
    for (int i=0; i<1000000; i++) {
        env.actions[0] = 2;
        c_step(&env);
    }
    printf("\tProfit: %f\n", env.log.profit / env.log.n);
    printf("\tExpense: %f\n", env.log.expense / env.log.n);
    printf("\tEpisode length: %f\n", env.log.episode_length / env.log.n);
    printf("\tEpisode return: %f\n", env.log.episode_return / env.log.n);
    printf("\tN: %f\n", env.log.n);
    //assert(env.agents[0].job_revenue == 950.0f && "Bilateral negotiation seller incorrect revenue");
    //assert(env.agents[1].job_revenue == 1000.0f && "Bilateral negotiation buyer incorrect revenue");
    //assert(env.agents[1].compute_expense == 950.0f && "Bilateral negotiation buyer incorrect expense");
    c_step(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    printf("Finished mirrored training environment\n\n");

    // Selling 200x gpu_2 test
    printf("Selling 200x single gpu_2 nodes\n");
    env = create_train_env();
    env.scripted_buy_price=1.0,
    env.scripted_buy_price_dr=0.0,
    env.job_duration=100;
    env.job_duration_dr=0;
    env.job_nodes[0] = 0;
    env.job_nodes[1] = 0;
    env.job_nodes[2] = 200;
    env.job_nodes_dr[2] = 0.0;
    env.job_tb_usage=1.0;
    env.job_tb_usage_dr=0.0;
 
    seller_spec = (ClusterSpec){
        .node_capacity = {0, 0, 200},
        .node_capacity_dr = {0.0, 0.0, 0.0},
        .tb_capacity = 200,
        .tb_capacity_dr = 0.0,
        .kwh_capacity = 0,
        .kwh_capacity_dr = 0.0,
        .kw_generation = 0,
        .kw_generation_dr = 0.0,
    };
    buyer_spec = (ClusterSpec){0};
    num_agents = env.ai_buyers + env.ai_sellers;
    init(&env, buyer_spec, seller_spec);
    env.observations = (float*)calloc(num_agents*env.num_obs, sizeof(float));
    env.actions = (int*)calloc(num_agents*NUM_ACT, sizeof(int));
    env.rewards = (float*)calloc(num_agents, sizeof(float));
    env.terminals = (unsigned char*)calloc(num_agents, sizeof(unsigned char));
    c_reset(&env);

    while (env.terminals[0] == 0) {
        env.actions[0] = 4;
        c_step(&env);
    }
    printf("\tProfit: %f\n", env.log.profit / env.log.n);
    printf("\tExpense: %f\n", env.log.expense / env.log.n);
    printf("\tEpisode length: %f\n", env.log.episode_length / env.log.n);
    printf("\tEpisode return: %f\n", env.log.episode_return / env.log.n);
    printf("\tN: %f\n", env.log.n);
    c_step(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    printf("Finished 200x single gpu_2 nodes\n");
}
