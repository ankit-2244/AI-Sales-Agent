const UPSTREAM = process.env.SALES_API_URL || "https://ai-sales-agent.fastapicloud.dev";

exports.handler = async () => {
  try {
    const resp = await fetch(UPSTREAM.replace(/\/$/, "") + "/health");
    const text = await resp.text();
    return {
      statusCode: resp.status,
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
      body: text,
    };
  } catch (err) {
    return {
      statusCode: 502,
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" },
      body: JSON.stringify({ status: "down", detail: err.message }),
    };
  }
};
