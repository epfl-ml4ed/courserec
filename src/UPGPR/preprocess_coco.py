import os
import argparse
import pickle

import pandas as pd


def read_courses(args):
    course_latest = pd.read_csv(
        os.path.join(args.dataset, "course_latest.csv"), encoding="utf-8"
    )

    # Remove courses not in english
    course_latest = course_latest[
        (course_latest.language == "english")
        & (course_latest.second_level_category != "other")
    ]
    return course_latest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/coco/coco")
    parser.add_argument("--save_dir", type=str, default="data/coco/processed_files")
    parser.add_argument("--min_concept_count", type=int, default=10)
    parser.add_argument("--min_user_count", type=int, default=10)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    course_latest = read_courses(args)

    teach_latest = pd.read_csv(
        os.path.join(args.dataset, "teach_latest.csv"), encoding="utf-8"
    )

    # Remove teachers teaching less than 2 courses
    teach_latest = teach_latest[
        teach_latest.course_id.isin(course_latest.course_id.unique())
    ]

    teach_latest = teach_latest.groupby("course_id").first().reset_index()

    instructors_min_two_courses = teach_latest[
        teach_latest.groupby("instructor_id")["instructor_id"].transform("size") > 2
    ]

    instructor_ids = instructors_min_two_courses.instructor_id.unique()

    courses_of_valid_instructors = teach_latest[
        teach_latest.instructor_id.isin(instructor_ids)
    ]

    course_ids = courses_of_valid_instructors.course_id.unique()

    valid_courses = course_latest[(course_latest.course_id.isin(course_ids))]

    evaluate_latest = pd.read_csv(
        os.path.join(args.dataset, "evaluate_latest.csv"),
        encoding="utf-8",
        index_col=0,
        # low_memory=False,
    )

    student_enrolments = evaluate_latest[
        evaluate_latest.course_id.isin(valid_courses.course_id)
    ]
    student_enrolments = student_enrolments[
        student_enrolments.groupby("learner_id")["learner_id"].transform("size") >= 10
    ]

    course_skill_df = pd.read_csv(
        os.path.join(args.dataset, "course_skill.csv"), encoding="utf-8"
    )
    learner_skill_df = pd.read_csv(
        os.path.join(args.dataset, "learner_skill.csv"), encoding="utf-8"
    )
    learners_with_skills = learner_skill_df.learner_id.unique()

    student_enrolments = student_enrolments[
        student_enrolments.learner_id.isin(learners_with_skills)
    ]

    instructor_id_to_num = {}
    with open(
        os.path.join(args.save_dir, "instructors.txt"), "w", encoding="utf-8"
    ) as f:
        for i, instr in enumerate(instructor_ids):
            f.write(str(instr) + "\n")
            instructor_id_to_num[instr] = i

    course_id_to_num = {}
    with open(os.path.join(args.save_dir, "courses.txt"), "w", encoding="utf-8") as f:
        for i, course in enumerate(valid_courses.course_id):
            f.write(str(int(course)) + "\n")
            course_id_to_num[course] = i

    courses_info = {}
    course_num_to_id = {}
    instructor_list = list(instructor_ids)

    for idx, c in enumerate(valid_courses.course_id):
        course_id = int(c)
        course_num = idx
        course_second_level_cat = course_latest[course_latest.course_id == course_id][
            "second_level_category"
        ].iloc[0]
        is_valid_instructor = False
        idx_instr = 0
        while is_valid_instructor == False:
            course_instr = teach_latest.loc[teach_latest.course_id == course_id].iloc[
                idx_instr
            ]["instructor_id"]
            idx_instr += 1
            if course_instr in instructor_list:
                is_valid_instructor = True
        # print(str(course_instr) + "\n")
        courses_info[course_id] = {
            "num": course_num,
            "s_category": course_second_level_cat,
            "instructor": course_instr,
        }
        course_num_to_id[idx] = course_id

    categories = (
        course_latest.groupby(["first_level_category", "second_level_category"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )

    categories.second_level_category.value_counts()

    valid_categories = categories.loc[categories.second_level_category != "other"]
    s_level_categories = valid_categories.second_level_category.unique()
    f_level_categories = valid_categories.first_level_category.unique()

    f_level_category_to_num = {}
    num_to_f_level_category = {}
    for i, l in enumerate(f_level_categories):
        f_level_category_to_num[l] = i
        num_to_f_level_category[i] = l

    s_level_category_to_num = {}
    num_to_s_level_category = {}
    for i, s in enumerate(s_level_categories):
        s_level_category_to_num[s] = i
        num_to_s_level_category[i] = s

    with open(
        os.path.join(args.save_dir, "second_categories.txt"), "w", encoding="utf-8"
    ) as f:
        for i in range(len(s_level_categories)):
            f.write(num_to_s_level_category[i] + "\n")

    with open(
        os.path.join(args.save_dir, "first_categories.txt"), "w", encoding="utf-8"
    ) as f:
        for i in range(len(f_level_categories)):
            f.write(num_to_f_level_category[i] + "\n")

    student_id_to_num = {}
    num_to_student_id = {}
    for i, learner_id in enumerate(student_enrolments.learner_id.unique()):
        student_id_to_num[learner_id] = i
        num_to_student_id[i] = learner_id
    with open(os.path.join(args.save_dir, "learners.txt"), "w", encoding="utf-8") as f:
        for i in range(len(student_id_to_num)):
            f.write(str(num_to_student_id[i]) + "\n")
    files = [
        "course_instructor.txt",
        "course_scategory.txt",
        "scategory_fcategory.txt",
        "enrolments.txt",
    ]
    for f in files:
        f = os.path.join(args.save_dir, f)
        if os.path.exists(f):
            os.remove(f)

    for i in range(len(valid_courses.course_id)):
        course_id = course_num_to_id[i]
        course_info = courses_info[course_id]
        with open(
            os.path.join(args.save_dir, "course_instructor.txt"), "a", encoding="utf-8"
        ) as f:
            f.write(str(instructor_id_to_num[course_info["instructor"]]) + "\n")
        with open(
            os.path.join(args.save_dir, "course_scategory.txt"), "a", encoding="utf-8"
        ) as f:
            f.write(str(s_level_category_to_num[course_info["s_category"]]) + "\n")

    for i in range(len(s_level_categories)):
        s_cat = num_to_s_level_category[i]
        f_cat = valid_categories.loc[
            valid_categories.second_level_category == s_cat
        ].iloc[0]["first_level_category"]
        f_cat_num = f_level_category_to_num[f_cat]
        with open(
            os.path.join(args.save_dir, "scategory_fcategory.txt"),
            "a",
            encoding="utf-8",
        ) as f:
            f.write(str(f_cat_num) + "\n")

    enrol_dict = {}
    with open(
        os.path.join(args.save_dir, "enrolments.txt"), "a", encoding="utf-8"
    ) as f:
        for idx in student_enrolments.index:
            learner_id = student_enrolments.learner_id[idx]
            learner_id = student_id_to_num[learner_id]
            if learner_id not in enrol_dict:
                enrol_dict[learner_id] = []
            course_id = int(student_enrolments.course_id[idx])
            course_id = course_id_to_num[course_id]
            enrol_dict[learner_id].append(course_id)
            f.write(f"{learner_id} {course_id}\n")

    pickle.dump(enrol_dict, open(os.path.join(args.save_dir, "enrolments.pkl"), "wb"))
    files = ["course_skills.txt", "skills.txt"]
    for f in files:
        f = os.path.join(args.save_dir, f)
        if os.path.exists(f):
            os.remove(f)

    course_skills_dict = {}
    skills = set()
    skill_id_to_num = {}
    for idx in course_skill_df.index:
        course_id = course_skill_df.course_id[idx]
        skill = course_skill_df.skills[idx]
        course_skills_dict[course_id] = course_skills_dict.get(course_id, []) + [skill]
        skills.add(skill)

    with open(os.path.join(args.save_dir, "skills.txt"), "a") as f:
        for i, s in enumerate(skills):
            skill_id_to_num[s] = i
            f.write(f"{s}\n")

    for i in range(len(valid_courses.course_id)):
        course_id = course_num_to_id[i]
        course_skills = course_skills_dict.get(course_id, [])
        skill_nums = []
        for cs in course_skills:
            skill_nums.append(str(skill_id_to_num[cs]))

        with open(
            os.path.join(args.save_dir, "course_skills.txt"), "a", encoding="utf-8"
        ) as f:
            f.write(" ".join(skill_nums) + "\n")

    learner_skills_dict = {}
    # remove all rows where learner-ID is not in student_id_to_num
    learner_skill_df = learner_skill_df[
        learner_skill_df.learner_id.isin(student_id_to_num.keys())
    ]

    for idx in learner_skill_df.index:
        l = learner_skill_df.learner_id[idx]
        sk = learner_skill_df.skills[idx]
        learner_skills_dict[l] = learner_skills_dict.get(l, []) + [sk]
    with open(os.path.join(args.save_dir, "learner_skills.txt"), "w") as f:
        for i in range(len(student_id_to_num)):
            l_id = num_to_student_id[i]

            l_skill_nums = [
                str(skill_id_to_num[l_skill]) for l_skill in learner_skills_dict[l_id]
            ]

            f.write(" ".join(l_skill_nums) + "\n")


if __name__ == "__main__":
    main()
