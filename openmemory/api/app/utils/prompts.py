MEMORY_CATEGORIZATION_PROMPT = """Your task is to assign each piece of information (or "memory") to one or more of the following categories. Feel free to use multiple categories per item when appropriate.

- Personal: family, friends, home, hobbies, lifestyle
- Relationships: social network, significant others, colleagues
- Preferences: likes, dislikes, habits, favorite media
- Health: physical fitness, mental health, diet, sleep
- Travel: trips, commutes, favorite places, itineraries
- Work: job roles, companies, projects, promotions
- Education: courses, degrees, certifications, skills development
- Projects: to‑dos, milestones, deadlines, status updates
- AI, ML & Technology: infrastructure, algorithms, tools, research
- Technical Support: bug reports, error logs, fixes
- Finance: income, expenses, investments, billing
- Shopping: purchases, wishlists, returns, deliveries
- Legal: contracts, policies, regulations, privacy
- Entertainment: movies, music, games, books, events
- Messages: emails, SMS, alerts, reminders
- Customer Support: tickets, inquiries, resolutions
- Product Feedback: ratings, bug reports, feature requests
- News: articles, headlines, trending topics
- Organization: meetings, appointments, calendars
- Goals: ambitions, KPIs, long‑term objectives

Guidelines:
- Return only the categories under 'categories' key in the JSON format.
- If you cannot categorize the memory, return an empty list with key 'categories'.
- Don't limit yourself to the categories listed above only. Feel free to create new categories based on the memory. Make sure that it is a single phrase.
"""

FACT_RETRIEVAL_PROMPT = """You are an Intelligent Information Extractor, specialized in capturing valuable information from conversations. Your role is to extract ONLY significant and actionable information, avoiding generic conversational patterns.

ONLY Extract These Types of Information:

1. SIGNIFICANT BEHAVIORAL PATTERNS:
   - Specific communication structure requirements (format preferences, language needs)
   - Demonstrated domain expertise and specializations
   - Unique problem-solving methodologies
   - Specific work patterns and professional approaches
   - Technical skill sets and competencies

2. ACTIONABLE INSIGHTS & KNOWLEDGE:
   - Specific research findings with data points, percentages, or measurable outcomes
   - Detailed market analysis and consumer behavior insights
   - Strategic business recommendations with supporting rationale
   - Technical solutions and implementation details
   - Industry trends with specific examples or evidence
   - Product information with concrete specifications
   - Financial data and performance metrics
   - Customer insights with quantifiable results
   - Process improvements with measurable impact

DO NOT EXTRACT:
- Generic conversation starters or greetings
- Basic willingness to help or assist
- Simple offers to provide information
- Standard professional courtesy responses
- General role descriptions without specific expertise
- Vague suggestions without concrete details
- Common conversational patterns

QUALITY THRESHOLD:
Only extract information that would be valuable for future decision-making or reference. Each fact should contain specific, actionable details.

Example - GOOD Extraction:

Input: "Our Q3 analysis shows a 15% increase in customer satisfaction after implementing the new support workflow. The primary improvement areas were response time (reduced by 40%) and first-call resolution (improved by 25%). I always structure my analysis reports with executive summary, methodology, key findings, and actionable recommendations - this format helps stakeholders quickly grasp the business impact."

Output: {{
  "facts": [
    "Q3 customer satisfaction increased by 15% after new support workflow implementation",
    "Support response time reduced by 40% with new workflow implementation",
    "First-call resolution improved by 25% with new support workflow",
    "User structures analysis reports with executive summary, methodology, key findings, and actionable recommendations for stakeholder clarity"
  ]
}}

Example - NO Extraction (too generic):

Input: "Hello! How can I assist you today as Royal Unibrew's Qualitative Market Insights Researcher? Here are a few suggestions for what you might want to explore: 1. Would you like insights on recent consumer trends? 2. Are you interested in product concepts? Let me know what you'd like to focus on!"

Output: {{"facts": []}}

Additional Examples:

Input: Hi.
Output: {{"facts": []}}

Input: "Based on our latest consumer research across 12 European markets, we found that 73% of Gen Z consumers prioritize sustainability credentials when choosing beverage brands, with 45% willing to pay a 15-20% premium for verified sustainable packaging. The strongest correlation (r=0.78) was between environmental messaging authenticity and purchase intent."

Output: {{
  "facts": [
    "73% of Gen Z consumers prioritize sustainability credentials when choosing beverage brands across 12 European markets",
    "45% of Gen Z consumers willing to pay 15-20% premium for verified sustainable packaging",
    "Strong correlation (r=0.78) between environmental messaging authenticity and purchase intent for beverages",
    "User conducts multi-market consumer research with statistical analysis and correlation studies"
  ]
}}

Return the facts in JSON format as shown above.

Strict Guidelines:
- ONLY extract information with specific details, data, or unique insights
- Do NOT extract basic conversational elements or generic offers to help
- Focus on information that would be referenced later for decision-making
- Each fact must be independently valuable and actionable
- Avoid extracting role descriptions unless they reveal specific expertise areas
- If the conversation contains only generic exchanges, return an empty facts array
- Extract both concrete insights AND meaningful behavioral patterns, but be highly selective

Following is a conversation between the user and the assistant. Extract ONLY valuable and specific information as demonstrated in the examples above."""
