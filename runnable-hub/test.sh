# curl "http://localhost:8000/API" -d '{"url":"http://ifconfig.me"}' --header "Content-Type: application/json"

#curl "http://localhost:8000/SHELL" -d '{"run":"uptime"}' --header "Content-Type: application/json"


curl "http://localhost:8000/PROCESS" -d '{"jobs":[{"jobId": "test", "steps":[{"stepId":"testStep", "runnableCode":"SHELL_WORKER", "request":{"run":"uptime"}}]}]}' --header "Content-Type: application/json"