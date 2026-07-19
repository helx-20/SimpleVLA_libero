import os

def calculate_stage1_failurerate(log_file_path):
    """
    Calculate the failure rate of stage 1 from the log file.

    Args:
        log_file_path (str): Path to the log file.
    """
    results_per_suite = {"libero_10": {'success': 0, 'failure': 0}, "libero_goal": {'success': 0, 'failure': 0}, "libero_spatial": {'success': 0, 'failure': 0}, "libero_object": {'success': 0, 'failure': 0}}
    with open(log_file_path, 'r') as f:
        for line in f:
            if line.startswith("[w") and "success" in line:
                info = line.split(" ")
                suite_name = info[1].split("/")[0]
                success = int(info[3].split("/")[0])
                failure = int(info[3].split("/")[1]) - success
                results_per_suite[suite_name]['success'] += success
                results_per_suite[suite_name]['failure'] += failure
    
    mean_results = {"success": 0, "failure": 0}
    for suite_name, results in results_per_suite.items():
        total = results['success'] + results['failure']
        failure_rate = results['failure'] / total
        mean_results['success'] += results['success']
        mean_results['failure'] += results['failure']
        print(f"Suite: {suite_name}, Failure Rate: {failure_rate:.2%}")
    print(f"Mean Failure Rate: {mean_results['failure'] / (mean_results['success'] + mean_results['failure']):.2%}")

if __name__ == "__main__":
    log_file_path = "adversarial_training/logs/stage1_collect.log"
    calculate_stage1_failurerate(log_file_path)
    