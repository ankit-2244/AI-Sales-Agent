const UPSTREAM = process.env.SALES_API_URL || "https://ai-sales-agent.fastapicloud.dev";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Content-Type",
  "Access-Control-Allow-Methods": "POST,OPTIONS",
};

exports.handler = async (event) => {
  if (event.httpMethod === "OPTIONS") {
    return { statusCode: 204, headers: CORS, body: "" };
  }
  if (event.httpMethod !== "POST") {
    return { statusCode: 405, headers: CORS, body: JSON.stringify({ detail: "POST only" }) };
  }
  try {
    const resp = await fetch(UPSTREAM.replace(/\/$/, "") + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: event.body || "{}",
    });
    const text = await resp.text();
    return {
      statusCode: resp.status,
      headers: { "Content-Type": "application/json", ...CORS },
      body: text,
    };
  } catch (err) {
    return {
      statusCode: 502,
      headers: { "Content-Type": "application/json", ...CORS },
      body: JSON.stringify({ detail: err.message || "Upstream API failed" }),
    };
  }
};
