import streamlit as st

from app.agent.clinical_agent import (
    run_clinical_agent,
)


# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="ClinicaSense AI",
    page_icon="🩺",
    layout="wide",
)


# =========================================
# HEADER
# =========================================

st.title("🩺 ClinicaSense AI")

st.subheader(
    "Agentic AI Clinical Decision Support System"
)

st.info(
    "ClinicaSense AI provides AI-assisted clinical "
    "information for educational and decision-support "
    "purposes. It does not replace professional "
    "medical judgment."
)


# =========================================
# CLINICAL CASE INPUT
# =========================================

clinical_case = st.text_area(
    "Enter Clinical Case",
    height=200,
    placeholder=(
        "Example:\n"
        "What clinical considerations should be "
        "reviewed for persistent elevated blood pressure?"
    ),
)


# =========================================
# ANALYZE BUTTON
# =========================================

if st.button(
        "🔍 Analyze Clinical Case",
        type="primary",
):

    if not clinical_case.strip():

        st.warning(
            "Please enter a clinical case."
        )

    else:

        # =========================================
        # RUN CLINICAL AGENT
        # =========================================

        with st.spinner(
                "ClinicaSense AI is analyzing the case..."
        ):

            result = run_clinical_agent(
                clinical_case
            )


        # =========================================
        # EXTRACT RESULTS
        # =========================================

        safety = result.get(
            "safety",
            {}
        )

        intent = result.get(
            "intent",
            "UNKNOWN"
        )

        clinical_domain = result.get(
            "clinical_domain",
            "UNKNOWN"
        )

        output_safe = result.get(
            "output_safe",
            False
        )

        safety_message = result.get(
            "safety_message",
            ""
        )


        # =========================================
        # AGENT WORKFLOW
        # =========================================

        st.divider()

        with st.expander(
                "🔎 View ClinicaSense AI Agent Workflow",
                expanded=True,
        ):

            # -----------------------------------------
            # STEP 1: INPUT
            # -----------------------------------------

            st.write(
                "🟢 **Input received**"
            )


            # -----------------------------------------
            # STEP 2: SAFETY SCREENING
            # -----------------------------------------

            st.write(
                "🟢 **Safety screening completed**"
            )


            # -----------------------------------------
            # STEP 3: INTENT ROUTING
            # -----------------------------------------

            if intent == "HIGH_RISK":

                st.warning(
                    "🚨 **Intent:** HIGH RISK"
                )

            elif intent == "CLINICAL_QUESTION":

                st.write(
                    "🧭 **Intent:** CLINICAL QUESTION"
                )

            else:

                st.write(
                    f"🧭 **Intent:** {intent}"
                )


            # -----------------------------------------
            # HIGH-RISK WORKFLOW
            # -----------------------------------------

            if intent == "HIGH_RISK":

                st.error(
                    "🛑 **High-risk case detected**"
                )

                st.write(
                    "🛡️ **Case routed directly to "
                    "Safety Response**"
                )

                st.write(
                    "⏹️ **Normal RAG and LLM clinical "
                    "reasoning workflow stopped**"
                )


            # -----------------------------------------
            # NORMAL CLINICAL WORKFLOW
            # -----------------------------------------

            else:

                # Clinical Domain Router
                st.write(
                    f"🩺 **Clinical Domain:** "
                    f"{clinical_domain}"
                )


                # Domain-aware RAG
                st.write(
                    f"🟢 **Medical RAG evidence retrieved "
                    f"for {clinical_domain} domain**"
                )


                # Prompt Builder
                st.write(
                    "🟢 **Evidence-grounded prompt created**"
                )


                # LLM
                st.write(
                    "🟢 **Gemini clinical reasoning completed**"
                )


                # Output Guardrail
                if output_safe:

                    st.success(
                        "🟢 **Output safety validation passed**"
                    )

                else:

                    st.error(
                        "🔴 **Output safety validation "
                        "requires review**"
                    )

                    if safety_message:

                        st.warning(
                            safety_message
                        )


        # =========================================
        # SAFETY RESULTS
        # =========================================

        st.divider()

        st.subheader(
            "🛡️ Safety Screening"
        )

        if safety.get(
                "requires_urgent_attention",
                False
        ):

            st.error(
                f"🚨 **HIGH-RISK ALERT**\n\n"
                f"{safety.get('message', '')}"
            )

        else:

            st.success(
                "✅ Initial safety screening completed."
            )


        # =========================================
        # CLINICAL DOMAIN
        # =========================================

        if intent != "HIGH_RISK":

            st.divider()

            st.subheader(
                "🩺 Clinical Domain Classification"
            )

            st.info(
                f"ClinicaSense AI classified this "
                f"case under: **{clinical_domain}**"
            )


        # =========================================
        # CLINICAL DECISION SUPPORT
        # =========================================

        st.divider()

        st.subheader(
            "🧠 Clinical Decision Support"
        )

        response = result.get(
            "response",
            ""
        )

        if response:

            st.markdown(
                response
            )

        else:

            st.warning(
                "No clinical decision-support response "
                "was generated."
            )


        # =========================================
        # RETRIEVED CLINICAL EVIDENCE
        # =========================================

        st.divider()

        st.subheader(
            "📚 Retrieved Clinical Evidence"
        )

        evidence_list = result.get(
            "evidence",
            []
        )

        if evidence_list:

            for index, evidence in enumerate(
                    evidence_list,
                    start=1,
            ):

                with st.expander(
                        f"Evidence {index}",
                        expanded=(index == 1),
                ):

                    st.write(
                        evidence
                    )

        else:

            if intent == "HIGH_RISK":

                st.info(
                    "Medical RAG was not executed because "
                    "the case was classified as high risk "
                    "and routed directly to the safety response."
                )

            else:

                st.info(
                    "No clinical evidence was retrieved "
                    "from the knowledge base."
                )


        # =========================================
        # EVIDENCE SOURCES
        # =========================================

        st.subheader(
            "📄 Evidence Sources"
        )

        sources = result.get(
            "sources",
            []
        )

        if sources:

            for source in sources:

                st.write(
                    f"• {source}"
                )

        else:

            if intent == "HIGH_RISK":

                st.info(
                    "No evidence sources were retrieved "
                    "because the case was routed directly "
                    "to the safety response."
                )

            else:

                st.info(
                    "No evidence sources were returned."
                )


        # =========================================
        # SAFETY DISCLAIMER
        # =========================================

        st.divider()

        st.caption(
            "⚠️ This is an AI-generated decision-support "
            "response based on retrieved knowledge and "
            "must not be used as a substitute for "
            "professional medical judgment."
        )