# Databricks notebook source
# 
# This file contains demographic feature retrieval functions used for the 
#   2022 Machine Learning Workshop (part of the Software Symposium)!
#   https://FORWARD_SITE/mlworkshop2022 
#      OR https://INFO_SITE/cdo/events/internal-events/4354c5db-3d3d-4481-97c4-8ad8f12686f1
#
# You can (and should) change them for your own experiments, but they are uniquely defined
# here for constants that we will use together.


# COMMAND ----------

# MAGIC %run ../utilities/settings

# COMMAND ----------

import geopandas as gpd
import pyspark.sql.functions as F
import pandas as pd


# COMMAND ----------

# load demographics from neustar/feature store, dropping personally identifiable field (e.g. name, custloc, etc)
def demographic_load_fs(list_columns=['zipcd', 'age', 'hshld_income_cd', 'hshld_incme_grp', 'est_curr_home_value', 'exact_age', 'gnrt', 'lang', 'ethnc_grp', 'ethnic_sub_grp', 'gndr', 'latitude_cif', 'longitude_cif', 'marital_status_cif', 'building_area_1_cif', 'lot_size_square_feet_cif', 'market_value_land_cif', 'market_value_improvement_cif', 'number_of_children_in_living_unit_cif', 'number_of_units_cif', 'number_of_bedrooms_cif', 'year_built_cif', 'tax_amount_cif', 'head_of_household_cif', 'length_of_residence_cif', 'edctn', 'state_cif', 'city_cif'], auth_token=None):
    from featurestore import Client
    import os
    os.environ['no_proxy'] = 'atlantis-feature-store.SERVICE_SITE'
    client = Client('atlantis-feature-store.SERVICE_SITE:8080')

    # Token='Enter your PAT token' --- see the configure file
    if auth_token is None:   # if you don't want to use secrets, just get it here...
        auth_token = CREDENTIALS['credentials']['ATLANTIS']
    client.auth.set_auth_token(auth_token)

    project = client.projects.get('Neustar')
    fs = project.feature_sets.get('Neustar Demographics')
    data_reference = fs.retrieve()
    df = data_reference.as_spark_frame(spark)
    fn_log(f"[demographic_load_fs] Available columns... {df.columns}")

    # filter to specifically requested columns
    if list_columns is not None:
        return df.select(list_columns)
    return df

# testing functions to load/save demographics
if False: 
    path_write = CREDENTIALS['paths']['demographics_raw']
    dbutils.fs.rm(path_write, True)
    df_sub = demographic_load_fs().filter(F.col('state_cif') == F.lit('NY'))
    df_sub.write.format('delta').save(path_write)
    df_sub = spark.read.format('delta').load(path_write)
    
    # columns at time of writing.... 
    # ['individual_key_cif', 'surname_cif', 'given_name_cif', 'middle_initial_cif', 'generational_suffix_cif', 'household_id_cif', 'primary_street_number_cif', 'primary_street_pre_dir_abbrev_cif', 'primary_street_name_cif', 'primary_street_suffix_cif', 'primary_street_post_dir_abbrev_cif', 'secondary_address_type_cif', 'secondary_address_number_cif', 'zip_code_cif', 'zip_plus4_cif', 'dpc_cif', 'usps_zip4_types_cif', 'crte_cif', 'city_cif', 'state_cif', 'dpv_validation_code_cif', 'do_not_mail_flag_cif', 'fips_numeric_state_code_cif', 'fips_county_cif', 'census_tract_cif', 'census_block_group_cif', 'census_block_id_cif', 'cbsa_cif', 'latlong_match_level_cif', 'latitude_cif', 'longitude_cif', 'rbdi_cif', 'time_zone_cif', 'daylight_saving_observed_cif', 'email_address_1_cif', 'phone_1_cif', 'phone_1_type_cif', 'phone_1_activity_status_cif', 'phone_1_prepaid_indicator_cif', 'phone_2_cif', 'phone_2_type_cif', 'phone_2_activity_status_cif', 'phone_2_prepaid_indicator_cif', 'phone_3_cif', 'phone_3_type_cif', 'phone_3_activity_status_cif', 'phone_3_prepaid_indicator_cif', 'e1_segment_cif', 'e1_segment_match_flag_cif', 'gender_cif', 'mob_cif', 'exact_age_cif', 'estimated_age_cif', 'marital_status_cif', 'education_model_cif', 'occupation_group_cif', 'occupation_code_cif', 'business_owner_cif', 'ethnic_code_cif', 'ethnic_group_cif', 'country_of_origin_cif', 'religion_cif', 'language_preference_cif', 'est_household_income_cif', 'number_of_children_in_living_unit_cif', 'children_presence_of_child_0_18_cif', 'children_age_0_3_cif', 'children_age_0_3_score_cif', 'children_age_0_3_gender_cif', 'children_age_4_6_cif', 'children_age_4_6_score_cif', 'children_age_4_6_gender_cif', 'children_age_7_9_cif', 'children_age_7_9_score_cif', 'children_age_7_9_gender_cif', 'children_age_10_12_cif', 'children_age_10_12_score_cif', 'children_age_10_12_gender_cif', 'children_age_13_15_cif', 'children_age_13_15_score_cif', 'children_age_13_15_gender_cif', 'children_age_16_18_cif', 'children_age_16_18_score_cif', 'children_age_16_18_gender_cif', 'length_of_residence_cif', 'dwelling_unit_size_cif', 'dwelling_type_cif', 'homeowner_probability_model_cif', 'homeowner_combined_homeownerrenter_cif', 'home_business_cif', 'propertyrealty_home_year_built_cif', 'estimated_current_home_value_cif', 'investment_property_zip_code_cif', 'auto_in_the_market_new_cif', 'auto_in_the_market_used_cif', 'auto_in_the_market_used_0_5_vehicle_cif', 'auto_in_the_market_used_6_10_vehicle_cif', 'auto_in_the_market_used_11_vehicle_cif', 'mail_responder_cif', 'upscale_merchandise_buyer_cif', 'male_merchandise_buyer_cif', 'female_merchandise_buyer_cif', 'craftshobby_merchandise_buyer_cif', 'gardeningfarming_buyer_cif', 'book_buyer_cif', 'collect_special_foods_buyer_cif', 'gifts_and_gadgets_buyer_cif', 'general_merchandise_buyer_cif', 'family_general_magazine_cif', 'female_oriented_magazine_cif', 'sports_magazine_cif', 'religious_magazine_cif', 'gardening_farming_magazine_cif', 'culinary_interests_magazine_cif', 'health_and_fitness_magazine_cif', 'do_it_yourselfers_cif', 'news_and_financial_cif', 'photography_cif', 'opportunity_seekers_ce_cif', 'religious_contributor_cif', 'political_contributor_cif', 'health_institution_contributor_cif', 'assessors_parcel_number_apn_pin_cif', 'duplicate_apn_cif', 'property_address_source_flag_cif', 'assessee_owner_name_cif', 'second_assessee_owner_name_or_dba_cif', 'assessee_owner_vesting_code_cif', 'tax_account_number_cif', 'mail_care_of_name_cif', 'assessee_mail_full_street_address_cif', 'assessee_mail_city_name_cif', 'assessee_mail_state_code_cif', 'assessee_mail_zip_code_cif', 'assessee_mail_zip_4_cif', 'assessee_mail_secondary_address_type_cif', 'assessee_mail_secondary_address_number_cif', 'owner_occupied_residential_cif', 'assessed_land_value_cif', 'assessed_improvement_value_cif', 'total_assessed_value_cif', 'assessment_year_cif', 'california_homeowners_exemption_cif', 'tax_exemption_codes_cif', 'tax_rate_code_area_cif', 'recorders_document_number_from_assessment_cif', 'recorders_book_number_from_assessment_cif', 'recorders_page_number_from_assessment_cif', 'recording_date_from_assessment_cif', 'document_type_from_assessment_county_description_cif', 'sales_price_from_assessment_cif', 'sales_price_code_from_assessment_cif', 'prior_sale_date_cif', 'prior_sales_price_cif', 'prior_sales_price_code_cif', 'tax_source_cif', 'tax_amount_cif', 'tax_year_cif', 'tax_delinquent_year_cif', 'legal_brief_description_cif', 'legal_brief_description_full_cif', 'legal_lot_code_cif', 'legal_lot_number_cif', 'legal_land_lot_cif', 'legal_block_cif', 'legal_section_cif', 'legal_district_cif', 'legal_unit_cif', 'legal_city_township_municipality_cif', 'legal_subdivision_name_cif', 'legal_phase_number_cif', 'legal_tract_number_cif', 'legal_section_township_range_meridian_cif', 'legal_assessors_map_ref_cif', 'standardized_land_use_code_cif', 'zoning_cif', 'lot_size_or_area_cif', 'lot_size_area_unit_cif', 'original_lot_size_or_area_cif', 'building_area_cif', 'year_built_cif', 'no_of_buildings_cif', 'no_of_stories_cif', 'total_number_of_rooms_cif', 'number_of_units_cif', 'number_of_bedrooms_cif', 'number_of_baths_cif', 'number_of_partial_baths_cif', 'garage_type_parking_cif', 'garage_parking_nbr_of_cars_cif', 'pool_cif', 'market_value_land_cif', 'market_value_improvement_cif', 'total_market_value_cif', 'market_value_year_cif', 'building_class_cif', 'style_cif', 'type_construction_cif', 'exterior_walls_cif', 'foundation_cif', 'roof_cover_cif', 'heating_cif', 'air_conditioning_cif', 'elevator_cif', 'fireplace_cif', 'basement_cif', 'current_owner_name_cif', 'current_owner_mail_street_address_cif', 'current_owner_mailing_city_cif', 'current_owner_mailing_state_cif', 'current_owner_mailing_zip_cif', 'current_owner_mailing_zip4_cif', 'current_owner_secondary_address_type_cif', 'current_owner_secondary_address_number_cif', 'current_owner_mail_care_of_name_cif', 'assessment_source_file_date_tape_cut_cif', 'latest_sale_recording_date_cif', 'latest_sale_book_number_cif', 'latest_sale_page_number_cif', 'latest_sale_document_number_cif', 'latest_sale_document_type_code_cif', 'latest_sale_price_cif', 'latest_sale_price_code_cif', 'prior_sale_recording_date_cif', 'prior_sale_book_number_cif', 'prior_sale_page_number_cif', 'prior_sale_document_number_cif', 'prior_sale_document_type_code_cif', 'prior_sale_price_cif', 'prior_sale_price_code_cif', 'latest_valid_recording_date_cif', 'latest_valid_book_number_cif', 'latest_valid_page_number_cif', 'latest_valid_document_number_cif', 'latest_valid_document_type_code_cif', 'latest_valid_price_cif', 'latest_valid_price_code_cif', 'prior_valid_recording_date_cif', 'prior_valid_book_number_cif', 'prior_valid_page_number_cif', 'prior_valid_document_number_cif', 'prior_valid_document_type_code_cif', 'prior_valid_price_cif', 'prior_valid_price_code_cif', 'assessee_owner_name_indicator_cif', 'second_assessee_owner_name_indicator_cif', 'mail_care_of_name_indicator_cif', 'assessee_owner_name_type_cif', 'second_assessee_owner_name_type_cif', 'alt_old_apn_indicator_cif', 'certification_date_cif', 'lot_size_square_feet_cif', 'building_quality_cif', 'floor_cover_cif', 'nbr_of_plumbing_fixtures_cif', 'building_area_1_cif', 'building_area_1_indicator_cif', 'building_area_2_cif', 'building_area_2_indicator_cif', 'building_area_3_cif', 'building_area_3_indicator_cif', 'building_area_4_cif', 'building_area_4_indicator_cif', 'building_area_5_cif', 'building_area_5_indicator_cif', 'building_area_6_cif', 'building_area_6_indicator_cif', 'building_area_7_cif', 'building_area_7_indicator_cif', 'effective_year_built_cif', 'heating_fuel_type_cif', 'air_conditioning_type_cif', 'lot_size_acres_refer_to_rules_for_populating_field_nbr76_cif', 'mortgage_lender_name_cif', 'interior_walls_cif', 'school_tax_district_1_cif', 'school_tax_district_1_indicator_cif', 'school_tax_district_2_cif', 'school_tax_district_2_indicator_cif', 'school_tax_district_3_cif', 'school_tax_district_3_indicator_cif', 'site_influence_cif', 'amenities_cif', 'other_impr_building_indicator_1_cif', 'other_impr_building_indicator_2_cif', 'other_impr_building_indicator_3_cif', 'other_impr_building_indicator_4_cif', 'neighborhood_code_cif', 'condo_project_bldg_name_cif', 'other_impr_building_indicator_5_cif', 'amenities_2_see_field_nbr162_cif', 'other_impr_building_area_1_cif', 'other_impr_building_area_2_cif', 'other_impr_building_area_3_cif', 'other_impr_building_area_4_cif', 'other_impr_building_area_5_cif', 'other_rooms_cif', 'extra_features_1_area_cif', 'extra_features_1_indicator_cif', 'topography_cif', 'roof_type_cif', 'extra_features_2_area_cif', 'extra_features_2_indicator_cif', 'extra_features_3_area_cif', 'extra_features_3_indicator_cif', 'extra_features_4_area_cif', 'extra_features_4_indicator_cif', 'old_apn_cif', 'building_condition_cif', 'lot_size_frontage_feet_refer_to_rules_for_populating_field_nbr76_cif', 'lot_size_depth_feet_refer_to_rules_for_populating_field_nbr76_cif', 'comments_summary_of_building_cards_dr_records_cif', 'water_cif', 'sewer_cif', 'recording_date_cif', 'recorders_book_number_cif', 'recorders_page_number_cif', 'recorders_document_number_cif', 'document_type_code_cif', 'deed_assessors_parcel_number_apn_pin_cif', 'multiapn_flag_cif', 'partial_interest_transferred_cif', 'seller_first_name_and_middle_name_1_cif', 'seller_last_name_or_corporation_name_1_cif', 'seller_id_code_1_cif', 'seller_first_name_and_middle_name_2_cif', 'seller_last_name_or_corporation_name_2_cif', 'buyer_first_name_and_middle_name_1_cif', 'buyer_last_name_or_corporation_name_1_cif', 'buyer_id_code_1_cif', 'buyer_first_name_and_middle_name_2_cif', 'buyer_last_name_or_corporation_name_2_cif', 'buyer_vesting_code_cif', 'concurrent_td_document_number_or_book_page_cif', 'buyer_mail_city_cif', 'buyer_mail_state_cif', 'buyer_mail_zip_code_cif', 'buyer_mail_zip_4_cif', 'recorders_map_reference_cif', 'propertybuyer_mailing_address_code_cif', 'property_use_code_cif', 'original_date_of_contract_cif', 'sales_price_cif', 'sales_price_code_cif', 'city_transfer_tax_cif', 'county_transfer_tax_cif', 'total_transfer_tax_cif', 'concurrent_td_lender_name_beneficiary_cif', 'concurrent_td_lender_type_cif', 'concurrent_td_loan_amount_cif', 'concurrent_td_loan_type_cif', 'concurrent_td_type_financing_cif', 'concurrent_td_interest_rate_cif', 'concurrent_td_due_date_cif', 'concurrent_2nd_td_loan_amount_cif', 'buyer_mail_full_street_address_cif', 'buyer_mail_unit_type_cif', 'buyer_mail_unit_number_cif', 'buyer_mail_care_of_name_cif', 'title_company_name_cif', 'complete_legal_description_code_cif', 'adjustable_rate_rider_cif', 'adjustable_rate_index_cif', 'change_index_cif', 'rate_change_frequency_cif', 'interest_rate_not_greater_than_cif', 'interest_rate_not_less_than_cif', 'maximum_interest_rate_cif', 'interest_only_period_cif', 'fixedstep_conversion_rate_rider_cif', 'first_change_dateyear_conversion_rider_cif', 'first_change_datemonth_and_day_conversion_rider_cif', 'prepayment_rider_cif', 'prepayment_term_penalty_rider_cif', 'reoflag_cif', 'distressedsaleflag_cif', 'cape_age_pop_median_age_cif', 'cape_age_pop_percentage_017_cif', 'cape_age_pop_percentage_1899_cif', 'cape_age_pop_percentage_6599_cif', 'cape_ethnic_pop_percentage_white_only_cif', 'cape_ethnic_pop_percentage_black_only_cif', 'cape_ethnic_pop_percentage_asian_only_cif', 'cape_ethnic_pop_percentage_hispanic_cif', 'cape_density_persons_per_hh_for_pop_in_hh_cif', 'cape_hhsize_hh_average_household_size_cif', 'cape_typ_hh_percentage_married_couple_family_cif', 'cape_child_hh_percentage_with_persons_lt18_cif', 'cape_child_hh_percentage_marr_couple_famw_persons_lt18_cif', 'cape_child_hh_percentage_marr_couple_famwo_persons_lt18_cif', 'cape_lang_hh_percentage_spanish_speaking_cif', 'cape_educ_pop25_median_education_attained_cif', 'cape_homval_oohu_median_home_value_cif', 'cape_hustr_hu_percentage_mobile_home_cif', 'cape_built_hu_median_housing_unit_age_cif', 'cape_tenancy_occhu_percentage_owner_occupied_cif', 'cape_tenancy_occhu_percentage_renter_occupied_cif', 'cape_educ_ispsa_cif', 'cape_educ_ispsa_decile_cif', 'cape_inc_family_inc_state_decile_cif', 'cape_inc_hh_median_family_household_income_cif', 'household_composition_cif', 'census_ruralurban_county_size_code_cif', 'head_of_household_cif', 'snpsht_dt_cif', 'snpsht_period_bgn_dt_cif', 'snpsht_period_end_dt_cif', 'cust_loc_id', 'src_cust_loc_id', 'addr_loc_src_addr_key_id', 'svc_addr_loc_src_addr_key_id', 'cust_ind', 'phn', 'mobl', 'eml1', 'eml2', 'eml3', 'cloc_cust_nm', 'first_nm', 'last_nm', 'addr_ln_1_txt', 'addr_ln_2_txt', 'addr_ln_3_txt', 'cty_nm', 'st_cd', 'zipcd', 'zipcd_plus', 'svc_addr_ln_1_txt', 'svc_addr_ln_2_txt', 'svc_addr_ln_3_txt', 'svc_cty_nm', 'svc_st_cd', 'svc_zipcd', 'svc_zipcd_plus', 'dob', 'dob_ind', 'acct_sgmnt_cd', 'liability', 'phn_1_vrfctn_scr', 'phn_2_vrfctn_scr', 'addr_1_vrfctn_scr', 'addr_2_vrfctn_scr', 'eml_1_vrfctn_scr', 'eml_2_vrfctn_scr', 'eml_3_vrfctn_scr', 'ekey', 'hshld_id', 'hshld_id_mtch_ind', 'first_name_mtch_ind', 'last_name_mtch_ind', 'first_addr_vrfy_ind', 'first_phn_vrfy_ind', 'first_eml_vrfy_ind', 'no_of_chldn_lvng', 'chldn_prsnc_of_chld_0_18', 'chldn_age_0_3', 'chldn_age_0_3_scr', 'chldn_age_0_3_gndr', 'chldn_age_4_6', 'chldn_age_4_6_scr', 'chldn_age_4_6_gndr', 'chldn_age_7_9', 'chldn_age_7_9_scr', 'chldn_age_7_9_gndr', 'chldn_age_10_12', 'chldn_age_10_12_scr', 'chldn_age_10_12_gndr', 'chldn_age_13_15', 'chldn_age_13_15_scr', 'chldn_age_13_15_gndr', 'chldn_age_16_18', 'chldn_age_16_18_scr', 'chldn_age_16_18_gndr', 'hshld_cmpst', 'hshld_income_cd', 'hshld_incme_grp', 'hshld_incme_sub_grp', 'est_curr_home_value', 'cape_homval_oohu_medn_hm_val', 'dwlng_unit_size', 'dwlng_type', 'hmownr_combnd_hmownrrntr', 'prptyrlty_hm_yr_built', 'lnght_of_resd', 'mail_rspndr', 'hm_buss', 'cnss_trct', 'cnss_blk_grp', 'cbsa', 'cnss_ru_urbn_cnty_size_cd', 'cape_inc_medn_fam_hshld_incme', 'gndr', 'mob', 'age', 'exact_age', 'est_age', 'gnrt', 'ocptn_cd', 'ocptn_grp', 'mrtl_sts', 'ethnc_grp_cd', 'ethnc_grp', 'ethnic_sub_grp', 'rlgn', 'lang', 'lang_cd', 'edctn_model', 'edctn', 'coo', 'buss_ownr', 'e1_sgmnt', 'snpsht_dt', 'snpsht_period_bgn_dt', 'snpsht_period_end_dt', 'geohash_lat_lng']

# COMMAND ----------

def demographic_factors_count(df_src, list_factors, list_demographics=None):
    """Attempt to compute aggregates according to the provided `list_factors` and `list_demographics`.
           This is essentially a big grouping operation, but it pivots the data frame for generation.
       Returns a dataframe with [(factor1, demo1, value), ... (factor1, demoN, value) ... (factorM, demoN, value)]
    """
    if list_demographics is None:
        list_demographics = ['hshld_incme_grp',   # household income group
                            'gnrt',   # "generation" from age
                            'marital_status_cif',   # marital status of the individual
                            # 'hshld_income_cd',    # household income rough number (in thousands)
                            'ethnc_grp',    # textual ethnic group
                            'edctn',    # education text name
                            'gndr',    # gender classification
                            'ethnic_sub_grp',   # ethnic sub-group
                            ]
    df_result = None
    for act_demographic in list_demographics:
        list_group = list_factors + [act_demographic]
        fn_log(f"[demographic_grouping] Attempting group with these factors... {list_group}")
        df_counted = (df_src
            .groupBy(*list_group).agg(F.count(F.col(act_demographic)).alias('count'))
            .withColumnRenamed(act_demographic, 'value')
            .withColumn('factor', F.lit(act_demographic))
        )
        if df_result is None:
            df_result = df_counted
        else:
            df_result = df_result.union(df_counted)
    return df_result


# COMMAND ----------

# df_neustar = demographic_load_fs()
# display(df_neustar.limit(50))
