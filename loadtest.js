import http from 'k6/http';
import { check, sleep } from 'k6';

// Configuration - passed via environment variables
const DEFAULT_URL = __ENV.DEFAULT_URL;
const LMI_URL = __ENV.LMI_URL;
const DURATION = __ENV.DURATION || '30s';
const VUS = parseInt(__ENV.VUS || '50');

if (!DEFAULT_URL || !LMI_URL) {
  throw new Error('Missing URLs. Set DEFAULT_URL and LMI_URL environment variables.');
}

export const options = {
  stages: [
    { duration: DURATION, target: VUS },   // Ramp up
    { duration: DURATION, target: VUS },   // Sustain
    { duration: DURATION, target: 0 },     // Ramp down
  ],
  thresholds: {
    'http_req_duration{endpoint:default}': ['p(95)<2000'],
    'http_req_duration{endpoint:lmi}': ['p(95)<2000'],
    'http_req_failed{endpoint:default}': ['rate<0.1'],
    'http_req_failed{endpoint:lmi}': ['rate<0.1'],
  },
};

export default function () {
  // Test Default Lambda endpoint
  const defaultRes = http.get(DEFAULT_URL, {
    tags: { endpoint: 'default', name: 'default' }
  });
  check(defaultRes, {
    'default: status is 200': (r) => r.status === 200,
  });

  // Test LMI Lambda endpoint
  const lmiRes = http.get(LMI_URL, {
    tags: { endpoint: 'lmi', name: 'lmi' }
  });
  check(lmiRes, {
    'lmi: status is 200': (r) => r.status === 200,
  });

  sleep(0.1);
}
