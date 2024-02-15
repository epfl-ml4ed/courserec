import os
import argparse


def save_learners(datadir, savedir, name):
    """Save learners to recbole format

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
    """
    users = []
    with open(os.path.join(datadir, "users.txt"), "r") as f:
        for line in f:
            users.append(line.strip())

    with open(os.path.join(savedir, name + ".user"), "w") as f:
        f.write("user_id:token\n")
        for user_id in users:
            f.write(f"{user_id}\n")
    return users


def save_courses(datadir, savedir, name):
    """Save coco courses to recbole format

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
    """

    courses = []
    with open(os.path.join(datadir, "courses.txt"), "r") as f:
        for line in f:
            courses.append(line.strip())

    with open(os.path.join(savedir, name + ".item"), "w") as f:
        f.write("item_id:token\n")
        for item_id in courses:
            f.write(f"{item_id}\n")
    return courses


def save_course_entity(savedir, name, courses):
    """Save coco course entity to recbole format

    Args:
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
        courses (list): list of courses
    """
    with open(os.path.join(savedir, name + ".link"), "w") as f:
        f.write("item_id:token\tentity_id:token\n")
        for course in courses:
            f.write(f"{course}\t{course}\n")


def read_concepts(datadir):
    """Read concepts from processed dataset

    Args:
        datadir (str): path of the processed dataset

    Returns:
        list: list of concepts
    """
    concepts = []
    with open(os.path.join(datadir, "concepts.txt"), "r") as f:
        for line in f:
            concepts.append(line.strip())

    return concepts


def read_schools(datadir):
    """Read schools from processed dataset

    Args:
        datadir (str): path of the processed dataset

    Returns:
        list: list of schools
    """
    schools = []
    with open(os.path.join(datadir, "schools.txt"), "r") as f:
        for line in f:
            schools.append(line.strip())

    return schools


def read_teachers(datadir):
    """Read teachers from processed dataset

    Args:
        datadir (str): path of the processed dataset

    Returns:
        list: list of teachers
    """
    teachers = []
    with open(os.path.join(datadir, "teachers.txt"), "r") as f:
        for line in f:
            teachers.append(line.strip())

    return teachers


def read_course_concepts(datadir, kg_triplets, courses, concepts):
    """Update kg triplets with concepts.

    Args:
        datadir (str): path of the processed dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
        concepts (list): list of concepts
    """
    with open(os.path.join(datadir, "course_concepts.txt"), "r") as f:
        for i, line in enumerate(f):
            course_concepts = line.strip()
            if course_concepts:
                for concept in course_concepts.split():
                    course = courses[int(i)]
                    kg_triplets.append(
                        [
                            course,
                            "has_concept",
                            concepts[int(concept)],
                        ]
                    )


def read_course_school(datadir, kg_triplets, courses, schools):
    """Update kg triplets with schools.

    Args:
        datadir (str): path of the dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
        schools (list): list of schools
    """
    with open(os.path.join(datadir, "course_school.txt"), "r") as f:
        for i, line in enumerate(f):
            course_schools = line.strip()
            if course_schools:
                for school in course_schools.split():
                    course = courses[int(i)]
                    kg_triplets.append(
                        [
                            course,
                            "in_school",
                            schools[int(school)],
                        ]
                    )


def read_course_teachers(datadir, kg_triplets, courses, teachers):
    """Update kg triplets with instructors.

    Args:
        datadir (str): path of the processed dataset
        kg_triplets (list): list of kg triplets
        courses (list): list of courses
        teacher (list): list of teacher
    """
    with open(os.path.join(datadir, "course_teachers.txt"), "r") as f:
        for i, line in enumerate(f):
            course_teachers = line.strip()
            if course_teachers:
                for teacher in course_teachers.split():
                    course = courses[int(i)]
                    kg_triplets.append(
                        [
                            course,
                            "has_teacher",
                            teachers[int(teacher)],
                        ]
                    )


def save_kg_triplets(kg_triplets, savedir, name):
    """Save kg_triplets to file.

    Args:
        kg_triplets (list): list of triplets as a tuple (head_id, relation_id, tail_id)
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
    """
    with open(os.path.join(savedir, name + ".kg"), "w") as f:
        f.write("head_id:token\trelation_id:token\ttail_id:token\n")
        for head_id, relation_id, tail_id in kg_triplets:
            f.write(f"{head_id}\t{relation_id}\t{tail_id}\n")


def save_enrolment(datadir, savedir, name, subset, learners, courses):
    """Save coco enrolments recbole format

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
        subset (str): name of the subset
        learners (list): list of learners
        courses (list): list of courses
    """
    enrolments = []
    with open(os.path.join(datadir, subset + ".txt"), "r") as f:
        for line in f:
            enrolments.append([int(x) for x in line.split()])

    with open(os.path.join(savedir, name + "." + subset + ".inter"), "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\n")
        for learner, course in enrolments:
            f.write(f"{learners[learner]}\t{courses[course]}\t1\n")


def format_pgpr_moocube(datadir, savedir, name):
    """Format processed dataset to recbole format

    Args:
        datadir (str): path of the processed dataset
        savedir (str): path of the recbole dataset
        name (str): name of the dataset
    """
    learners = save_learners(datadir, savedir, name)
    courses = save_courses(datadir, savedir, name)

    subsets = ["train", "validation", "test"]

    for subset in subsets:
        save_enrolment(datadir, savedir, name, subset, learners, courses)

    save_course_entity(savedir, name, courses)
    concepts = read_concepts(datadir)
    teachers = read_teachers(datadir)
    schools = read_schools(datadir)
    kg_triplets = []
    read_course_concepts(datadir, kg_triplets, courses, concepts)
    read_course_school(datadir, kg_triplets, courses, schools)
    read_course_teachers(datadir, kg_triplets, courses, teachers)
    save_kg_triplets(kg_triplets, savedir, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="data/mooc/processed_files")
    parser.add_argument("--savedir", type=str, default="data/mooc/recbolemoocube")
    parser.add_argument("--dataset_name", type=str, default="recbolemoocube")
    args = parser.parse_args()

    # Creates the folder savedir if it does not exist
    os.makedirs(args.savedir, exist_ok=True)

    format_pgpr_moocube(args.datadir, args.savedir, args.dataset_name)
