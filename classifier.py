# ============================================================
# POD CLASSIFIER
# Deterministic business rules
# ============================================================

CATEGORIES = {
    "CLEAN_POD_SEAL_AND_SIGNATURE",
    "CLEAN_POD_ONLY_SEAL",
    "CLEAN_POD_ONLY_SIGNATURE",
    "ISSUE_POD_DAMAGED",
    "ISSUE_POD_SHORT",
    "ISSUE_POD_DAMAGED_AND_SHORT",
    "NO_SIGNATURE_NO_STAMP",
    "MANUAL_CHECK_REQUIRED"
}


def classify_pod(evidence):
    """
    Convert Qwen's evidence into one of the 8 assessment categories.

    Qwen is responsible for visual/document understanding.
    This function is responsible for deterministic classification.
    """

    # --------------------------------------------------------
    # Read evidence safely
    # --------------------------------------------------------

    physical_damage = bool(
        evidence.get("physicalDamage", False)
    )

    business_damage = bool(
        evidence.get("businessDamage", False)
    )

    shortage = bool(
        evidence.get("shortage", False)
    )

    has_signature = bool(
        evidence.get("hasSignature", False)
    )

    has_stamp = bool(
        evidence.get("hasStamp", False)
    )

    # --------------------------------------------------------
    # PRIORITY 1
    # Physical POD paper damage
    # --------------------------------------------------------

    if physical_damage:

        return (
            "MANUAL_CHECK_REQUIRED",
            "The physical POD document is damaged."
        )

    # --------------------------------------------------------
    # PRIORITY 2
    # Business damage + shortage
    # --------------------------------------------------------

    if business_damage and shortage:

        return (
            "ISSUE_POD_DAMAGED_AND_SHORT",
            "The POD indicates both goods/material damage and shortage."
        )

    # --------------------------------------------------------
    # PRIORITY 3
    # Business damage
    # --------------------------------------------------------

    if business_damage:

        return (
            "ISSUE_POD_DAMAGED",
            "The POD indicates that the goods/material were damaged."
        )

    # --------------------------------------------------------
    # PRIORITY 4
    # Shortage
    # --------------------------------------------------------

    if shortage:

        return (
            "ISSUE_POD_SHORT",
            "The POD indicates that goods/material/quantity were received short."
        )

    # --------------------------------------------------------
    # PRIORITY 5
    # Signature + stamp
    # --------------------------------------------------------

    if has_signature and has_stamp:

        return (
            "CLEAN_POD_SEAL_AND_SIGNATURE",
            "The POD contains both a recognizable signature and stamp/seal."
        )

    # --------------------------------------------------------
    # PRIORITY 6
    # Stamp only
    # --------------------------------------------------------

    if has_stamp:

        return (
            "CLEAN_POD_ONLY_SEAL",
            "The POD contains a recognizable stamp/seal but no recognizable signature."
        )

    # --------------------------------------------------------
    # PRIORITY 7
    # Signature only
    # --------------------------------------------------------

    if has_signature:

        return (
            "CLEAN_POD_ONLY_SIGNATURE",
            "The POD contains a recognizable signature but no recognizable stamp/seal."
        )

    # --------------------------------------------------------
    # PRIORITY 8
    # No signature + no stamp
    # --------------------------------------------------------

    return (
        "NO_SIGNATURE_NO_STAMP",
        "No recognizable signature or stamp/seal was identified."
    )

if __name__ == "__main__":

    test_cases = [

        {
            "name": "Damaged + Short",
            "evidence": {
                "physicalDamage": False,
                "businessDamage": True,
                "shortage": True,
                "hasSignature": True,
                "hasStamp": True
            }
        },

        {
            "name": "Damaged",
            "evidence": {
                "physicalDamage": False,
                "businessDamage": True,
                "shortage": False,
                "hasSignature": True,
                "hasStamp": True
            }
        },

        {
            "name": "Short",
            "evidence": {
                "physicalDamage": False,
                "businessDamage": False,
                "shortage": True,
                "hasSignature": True,
                "hasStamp": True
            }
        },

        {
            "name": "Signature + Stamp",
            "evidence": {
                "physicalDamage": False,
                "businessDamage": False,
                "shortage": False,
                "hasSignature": True,
                "hasStamp": True
            }
        },

        {
            "name": "Stamp only",
            "evidence": {
                "physicalDamage": False,
                "businessDamage": False,
                "shortage": False,
                "hasSignature": False,
                "hasStamp": True
            }
        },

        {
            "name": "Signature only",
            "evidence": {
                "physicalDamage": False,
                "businessDamage": False,
                "shortage": False,
                "hasSignature": True,
                "hasStamp": False
            }
        },

        {
            "name": "Neither",
            "evidence": {
                "physicalDamage": False,
                "businessDamage": False,
                "shortage": False,
                "hasSignature": False,
                "hasStamp": False
            }
        },

        {
            "name": "Physical damage",
            "evidence": {
                "physicalDamage": True,
                "businessDamage": True,
                "shortage": True,
                "hasSignature": True,
                "hasStamp": True
            }
        }
    ]

    for case in test_cases:

        category, reason = classify_pod(
            case["evidence"]
        )

        print(f"\n{case['name']}")
        print(f"Category: {category}")
        print(f"Reason:   {reason}")