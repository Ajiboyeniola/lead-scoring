import pandas as pd
import numpy as np

def generate_leads(n=2000):
    np.random.seed(42)

    # --- Synthetic company name generator ---
    adjectives = [
        "Blue", "Nova", "Spark", "Crest", "Apex", "Orbit", "Luminary",
        "Sterling", "Beacon", "Catalyst", "Mosaic", "Elevate", "Zenith",
        "Horizon", "Ember", "Pulse", "Stride", "Vibe", "Summit", "Peak"
    ]

    nouns = [
        "Media", "Solutions", "Creative", "Digital", "Brands", "Studios",
        "Agency", "Group", "Marketing", "Communications", "Consulting",
        "Ventures", "Partners", "Labs", "Works", "Co", "Hub", "Tech",
        "Design", "Productions", "Services", "Collective", "Network", "Firm"
    ]

    unique_companies = list(set([
        f"{np.random.choice(adjectives)} {np.random.choice(nouns)}"
        for _ in range(1000)
    ]))[:500]

    company_assignments = np.random.choice(unique_companies, n)

    # --- Engagement score drives behavioral features ---
    engagement = np.random.beta(2, 6, n)

    # --- Categorical columns ---
    lead_source = np.random.choice(
        ["LinkedIn", "Referral", "Website",
         "Cold Outreach", "Event"], n
    )

    ad_platform = np.random.choice(
        ["LinkedIn", "Instagram", "Facebook",
         "TikTok", "X", "YouTube", "None"], n
    )

    industry = np.random.choice(
        [
            "Technology", "Healthcare", "Finance", "Legal",
            "Consulting", "Real Estate", "Education",
            "Fashion & Apparel", "Beauty & Wellness", "Food & Beverage",
            "Home Decor", "Jewelry & Accessories", "Photography",
            "Interior Design", "Fitness & Wellness", "Hair & Skincare",
            "Event Planning", "Catering", "Handmade & Crafts",
            "Retail", "E-commerce", "Thrift & Vintage",
            "Children & Baby Products", "Pet Products",
            "Entertainment", "Content Creation", "Music",
            "Art & Illustration", "Videography", "Podcasting",
            "Logistics", "Construction", "Non-Profit", "Agriculture"
        ], n
    )

    business_type = np.random.choice(
        ["Personal Brand", "Small Business", "SME", "Corporate"],
        n,
        p=[0.25, 0.35, 0.25, 0.15]
    )

    company_size = np.random.choice(
        ["Small", "Medium", "Large"], n,
        p=[0.6, 0.3, 0.1]
    )

    job_title = np.random.choice(
        ["CEO", "Founder", "Co-Founder", "Owner",
         "Managing Director", "Director", "VP of Marketing",
         "VP of Sales", "Marketing Manager", "Sales Manager",
         "Business Development Manager", "Operations Manager",
         "General Manager", "Executive", "Consultant"], n
    )

    funnel_stage = np.random.choice(
        ["Awareness", "Consideration", "Decision"], n,
        p=[0.5, 0.3, 0.2]
    )

    preferred_contact_time = np.random.choice(
        ["Morning", "Afternoon", "Evening"], n
    )

    # --- Binary columns driven by engagement ---
    demo_requested = (
        engagement + np.random.normal(0, 0.2, n) > 0.6
    ).astype(int)

    webinar_attended = (
        engagement + np.random.normal(0, 0.2, n) > 0.65
    ).astype(int)

    budget_indicated = (
        engagement + np.random.normal(0, 0.2, n) > 0.55
    ).astype(int)

    decision_maker = np.random.choice(
        [0, 1], n, p=[0.65, 0.35]
    )

    # --- New Tier 1 features driven by engagement ---

    # Email reply rate — engaged leads reply more
    email_reply_rate = np.clip(
        engagement + np.random.normal(0, 0.15, n), 0, 1
    ).round(2)

    # Response time — engaged leads respond faster
    response_time_days = np.clip(
        (1 - engagement) * 14 + np.random.exponential(2, n), 0.5, 30
    ).round(1)

    # Interactions in last 30 days — engaged leads more active recently
    interactions_last_30_days = np.clip(
        np.random.poisson(engagement * 10 + 1, n), 0, 20
    )

    # Pricing page visited — engaged leads visit high intent pages
    pricing_page_visited = (
        engagement + np.random.normal(0, 0.2, n) > 0.6
    ).astype(int)

    # Meeting held — driven by engagement + sales follow up
    meeting_held = (
        engagement + np.random.normal(0, 0.15, n) > 0.65
    ).astype(int)

    # Proposal sent — only happens after meeting
    proposal_sent = np.where(
        meeting_held == 1,
        (engagement + np.random.normal(0, 0.2, n) > 0.55).astype(int),
        0
    )

    # --- Behavioral columns driven by engagement ---
    website_visits = np.clip(
        np.random.poisson(engagement * 30 + 1, n), 0, 50
    )

    pages_viewed = np.clip(
        website_visits + np.random.randint(0, 5, n), 1, 60
    )

    email_opens = np.clip(
        np.random.poisson(engagement * 20 + 1, n), 0, 25
    )

    email_opens = np.where(
        webinar_attended == 1,
        np.clip(email_opens + np.random.randint(2, 8, n), 0, 25),
        email_opens
    )

    email_clicks = np.clip(
        (email_opens * 0.6).astype(int) + np.random.randint(0, 3, n),
        0, 15
    )

    content_downloads = np.clip(
        np.random.poisson(engagement * 8 + 0.5, n), 0, 10
    )

    ad_clicks = np.clip(
        np.random.poisson(engagement * 6 + 0.5, n), 0, 15
    )

    num_contacts = np.where(
        demo_requested == 1,
        np.random.randint(5, 20, n),
        np.random.randint(1, 10, n)
    )

    # Engaged leads contacted more recently
    days_since_last_contact = np.clip(
        np.random.poisson((1 - engagement) * 80 + 5, n), 1, 180
    )

    # --- Created date ---
    days_back = np.random.exponential(scale=200, size=n).astype(int)
    days_back = np.clip(days_back, 1, 730)
    created_date = pd.Timestamp.today() - pd.to_timedelta(days_back, unit='D')
    lead_age_days = (pd.Timestamp.today() - created_date).days

    # --- Funnel stage weights ---
    stage_weights = np.array([
        {"Awareness": 0, "Consideration": 0.10, "Decision": 0.20}[s]
        for s in funnel_stage
    ])

    # --- Conversion score using logit ---
    logit = (
        -3.0 +

        # Strong binary signals
        2.0 * demo_requested +
        1.5 * budget_indicated +
        1.2 * decision_maker +
        1.0 * webinar_attended +

        # New strong signals
        2.5 * proposal_sent +
        1.8 * meeting_held +
        1.5 * pricing_page_visited +
        1.2 * email_reply_rate +
        -0.8 * (response_time_days / 30) +

        # Funnel stage
        stage_weights * 2.0 +

        # Behavioral signals normalized
        0.8 * (website_visits / 50) +
        0.7 * (email_opens / 25) +
        0.6 * (email_clicks / 15) +
        0.5 * (content_downloads / 10) +
        0.4 * (pages_viewed / 60) +
        0.3 * (interactions_last_30_days / 20) +
        0.2 * (ad_clicks / 15) +

        # Recency
        -1.0 * (days_since_last_contact / 180) +

        # Low noise
        np.random.normal(0, 0.3, n)
    )

    # Convert to probability using sigmoid
    prob = 1 / (1 + np.exp(-logit))

    # Sample conversions
    converted = np.array([
        np.random.choice([0, 1], p=[1 - p, p]) for p in prob
    ])

    # --- Build dataframe ---
    df = pd.DataFrame({
        "lead_id": range(1, n + 1),
        "company_name": company_assignments,
        "business_type": business_type,
        "created_date": created_date,
        "lead_age_days": lead_age_days,
        "email_opens": email_opens,
        "email_clicks": email_clicks,
        "email_reply_rate": email_reply_rate,
        "response_time_days": response_time_days,
        "website_visits": website_visits,
        "pages_viewed": pages_viewed,
        "pricing_page_visited": pricing_page_visited,
        "demo_requested": demo_requested,
        "meeting_held": meeting_held,
        "proposal_sent": proposal_sent,
        "content_downloads": content_downloads,
        "webinar_attended": webinar_attended,
        "industry": industry,
        "company_size": company_size,
        "job_title": job_title,
        "funnel_stage": funnel_stage,
        "budget_indicated": budget_indicated,
        "decision_maker": decision_maker,
        "days_since_last_contact": days_since_last_contact,
        "interactions_last_30_days": interactions_last_30_days,
        "num_contacts": num_contacts,
        "lead_source": lead_source,
        "ad_clicks": ad_clicks,
        "ad_platform": ad_platform,
        "preferred_contact_time": preferred_contact_time,
        "converted": converted
    })

    # --- Inject missing values ---
    for col, rate in [
        ("budget_indicated", 0.35),
        ("days_since_last_contact", 0.03),
        ("ad_platform", 0.04),
        ("job_title", 0.30),
        ("email_reply_rate", 0.10),
        ("response_time_days", 0.10),
        ("pricing_page_visited", 0.05),
    ]:
        mask = np.random.rand(n) < rate
        df.loc[mask, col] = np.nan

    return df


# Generate and save
df = generate_leads(n=2000)
print(df.shape)
print(df['converted'].value_counts())
print(df.head())